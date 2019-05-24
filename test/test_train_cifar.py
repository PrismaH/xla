import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='/tmp/cifar-data',
    batch_size=128,
    num_epochs=20,
    momentum=0.9,
    lr=0.2,
    target_accuracy=95.0)

from common_utils import TestCase, run_tests
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import torchvision
import torchvision.transforms as transforms
import unittest

class shortcut(nn.Module):
    def __init__(self, add_ch, stride=1):
        super(shortcut, self).__init__()
        #should use nn.Identity() but causes error on some environment
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False) if stride == 2 else lambda x: x
        self.pad = (0, 0, 0, 0, 0, add_ch)
        
    def forward(self, x):
        return self.pooling(F.pad(x, self.pad))

#basic block of pyramidnet
class Basicblock(nn.Module): 

    def __init__(self, input_ch, output_ch, stride=1):
        super(Basicblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_ch)
        self.conv1 = nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_ch)
        self.conv2 = nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = shortcut(output_ch - input_ch, stride)

    def forward(self, x):
        shortcut = self.down_sample(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)

        return x + shortcut

class Bottleneckblock(nn.Module):
    outchannel_ratio = 4
    reduction = 16

    def __init__(self, input_ch, output_ch, stride=1):
        super(Bottleneckblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_ch)
        self.conv1 = nn.Conv2d(input_ch, output_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_ch)
        self.conv2 = nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_ch)
        self.conv3 = nn.Conv2d(output_ch, output_ch * Bottleneckblock.outchannel_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(output_ch * Bottleneckblock.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.downconv = nn.Conv2d(output_ch * Bottleneckblock.outchannel_ratio, output_ch * Bottleneckblock.outchannel_ratio // Bottleneckblock.reduction, kernel_size=1, bias=False)
        self.upconv = nn.Conv2d(output_ch * Bottleneckblock.outchannel_ratio // Bottleneckblock.reduction, output_ch * Bottleneckblock.outchannel_ratio, kernel_size=1, bias=False)
        self.down_sample = shortcut(output_ch * Bottleneckblock.outchannel_ratio - input_ch, stride)

    def forward(self, x):
        shortcut = self.down_sample(x)

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn4(x)

        y = self.global_pool(x)
        y = self.downconv(y)
        y = self.relu(y)
        y = self.upconv(y)
        y = self.sigmoid(y)

        x = x * y

        return x + shortcut

class SEPyramidNet(nn.Module): #Constructing PyramidNet
    def __init__(self, block_type, num_res, alpha, input_ch, classes_num):
        super(SEPyramidNet, self).__init__()
        self.in_channels = 16
        self.num_res = num_res
        self.addrate = alpha / (num_res * 3)

        self.conv1 = nn.Conv2d(in_channels=input_ch, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self.layer_define(block_type, stride = 1)
        self.layer2 = self.layer_define(block_type, stride = 2)
        self.layer3 = self.layer_define(block_type, stride = 2)

        self.output_ch = round(self.output_ch)

        self.bn2 = nn.BatchNorm2d(self.output_ch)
        
        self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(self.output_ch, classes_num)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def layer_define(self, block_type, stride):
        layers_list = []
        for _ in range(self.num_res - 1):
            self.output_ch = self.in_channels + self.addrate
            layers_list.append(block_type(int(round(self.in_channels)), int(round(self.output_ch)), stride=stride))
            self.in_channels = self.output_ch
            stride = 1
 
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)

        return self.linear1(x.view(x.size(0), -1))
   
def ResNet18():
  return SEPyramidNet(Basicblock, 18, 48, 3, 10)


def train_cifar():
  print('==> Preparing data..')

  if FLAGS.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, 32,
                          32), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=50000 // FLAGS.batch_size)
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, 32,
                          32), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=10000 // FLAGS.batch_size)
  else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=FLAGS.datadir,
        train=True,
        download=True,
        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=FLAGS.datadir,
        train=False,
        download=True,
        transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  devices = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  model_parallel = dp.DataParallel(ResNet18, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=5e-4)
    tracker = xm.RateTracker()

    for x, (data, target) in loader:
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(FLAGS.batch_size)
      if x % FLAGS.log_steps == 0:
        print('[{}]({}) Loss={:.5f} Rate={:.2f}'.format(device, x, loss.item(),
                                                        tracker.rate()))

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    for x, (data, target) in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    print('[{}] Accuracy={:.2f}%'.format(device,
                                         100.0 * correct / total_samples))
    return correct / total_samples

  accuracy = 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = sum(accuracies) / len(devices)
    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  return accuracy * 100.0


class TrainCIFAR10(TestCase):

  def tearDown(self):
    super(TrainCIFAR10, self).tearDown()
    if FLAGS.tidy and os.path.isdir(FLAGS.datadir):
      shutil.rmtree(FLAGS.datadir)

  def test_accurracy(self):
    self.assertGreaterEqual(train_cifar(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
