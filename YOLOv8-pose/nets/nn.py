import math

import torch

from utils.util import make_anchors


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class Bottleneck(torch.nn.Module):
    """Ultralytics-style bottleneck: 3×3 → 3×3, with optional residual."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1, g=g)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(torch.nn.Module):
    """Ultralytics-style C2f: 1×1 split → n×Bottleneck → concat → 1×1."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = torch.nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# Backward compatibility: keep CSP as alias for pickled checkpoints
CSP = C2f


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              C2f(width[2], width[2], depth[0], True)]
        p3 = [Conv(width[2], width[3], 3, 2),
              C2f(width[3], width[3], depth[1], True)]
        p4 = [Conv(width[3], width[4], 3, 2),
              C2f(width[4], width[4], depth[2], True)]
        p5 = [Conv(width[4], width[5], 3, 2),
              C2f(width[5], width[5], depth[0], True),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkNetP2(torch.nn.Module):
    """DarkNet variant that also outputs P2 (for small object detection)."""
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2), C2f(width[2], width[2], depth[0], True)]
        p3 = [Conv(width[2], width[3], 3, 2), C2f(width[3], width[3], depth[1], True)]
        p4 = [Conv(width[3], width[4], 3, 2), C2f(width[4], width[4], depth[2], True)]
        p5 = [Conv(width[4], width[5], 3, 2), C2f(width[5], width[5], depth[0], True), SPP(width[5], width[5])]
        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p2, p3, p4, p5  # P2, P3, P4, P5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = C2f(width[4] + width[5], width[4], depth[0], False)
        self.h2 = C2f(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = C2f(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = C2f(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class DarkFPN_P2(torch.nn.Module):
    """Full FPN for P2 input: top-down P5→P4→P3→P2 + bottom-up P2→P3→P4."""
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        # Top-down: P5→P4→P3→P2
        self.h1 = C2f(width[4] + width[5], width[4], depth[0], False)
        self.h2 = C2f(width[3] + width[4], width[3], depth[0], False)
        self.h0 = C2f(width[2] + width[3], width[2], depth[0], False)
        # Bottom-up: P2→P3→P4
        self.h3 = Conv(width[2], width[2], 3, 2)
        self.h4 = C2f(width[2] + width[3], width[3], depth[0], False)
        self.h5 = Conv(width[3], width[3], 3, 2)
        self.h6 = C2f(width[3] + width[4], width[4], depth[0], False)

    def forward(self, x):
        p2, p3, p4, p5 = x
        # Top-down
        n4 = self.h1(torch.cat([self.up(p5), p4], 1))
        n3 = self.h2(torch.cat([self.up(n4), p3], 1))
        n2 = self.h0(torch.cat([self.up(n3), p2], 1))
        # Bottom-up
        n3b = self.h4(torch.cat([self.h3(n2), n3], 1))
        n4b = self.h6(torch.cat([self.h5(n3b), n4], 1))
        return n3b, n4b  # P3, P4 (detection heads only)


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.kpt_shape = kpt_shape  # number of kpt, number of dims (2 for x,y or 3 for x,y,visible)
        self.num_kpt = kpt_shape[0] * kpt_shape[1]  # number of kpt total

        c1 = max((16, ch[0] // 4, self.ch * 4))
        c2 = max(ch[0], self.nc)
        c3 = max(ch[0] // 4, self.num_kpt)
        self.dfl = DFL(self.ch) if self.ch > 1 else torch.nn.Identity()
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, self.nc, 1)) for x in ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, 4 * self.ch, 1)) for x in ch)
        self.kpt = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c3, 3),
                                                           Conv(c3, c3, 3),
                                                           torch.nn.Conv2d(c3, self.num_kpt, 1)) for x in ch)

    def forward(self, x):
        x_det = self.detect_box(x)
        x_kpt = self.detect_kpt(x)
        if self.training:
            return x_det, x_kpt
        return torch.cat([x_det, self.decode_kpt(x_kpt)], dim=1)

    def detect_box(self, x):
        x_det = []
        shape = x[0].shape[0]
        for i in range(self.nl):
            x_det.append(torch.cat((self.box[i](x[i]), self.cls[i](x[i])), dim=1))
        if self.training:
            return x_det

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x_det, self.stride, 0.5))

        x_cat = torch.cat([i.view(shape, self.no, -1) for i in x_det], dim=2)
        box, cls = x_cat.split((self.ch * 4, self.nc), 1)
        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), dim=1)
        return torch.cat((box * self.strides, cls.sigmoid()), dim=1)

    def detect_kpt(self, x):
        x_kpt = []
        shape = x[0].shape[0]
        for i in range(self.nl):
            kpt = self.kpt[i](x[i])
            kpt = kpt.view(shape, self.num_kpt, -1)
            x_kpt.append(kpt)
        return torch.cat(x_kpt, dim=-1)

    def decode_kpt(self, x):
        y = x.clone()
        dim = self.kpt_shape[1]
        if dim == 3:
            y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::dim] = (y[:, 0::dim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::dim] = (y[:, 1::dim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y

    def initialize_biases(self):
        # Initialize Detect() biases,
        # WARNING: requires stride availability
        for a, b, s in zip(self.box, self.cls, self.stride):
            # box
            a[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes, kpt_shape=(17, 3)):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)
        self.use_fpn = True

        img_dummy = torch.zeros(1, 3, 256, 256)
        feats = self.fpn(self.net(img_dummy))
        self.head = Head(num_classes, kpt_shape, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in feats])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class YOLO_P3(torch.nn.Module):
    """Variant 3: P3-only with full FPN. FPN does P5→P4→P3, Head on P3 only."""
    def __init__(self, width, depth, num_classes, kpt_shape=(17, 3)):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)
        self.use_fpn = True

        img_dummy = torch.zeros(1, 3, 256, 256)
        h2, _, _ = self.fpn(self.net(img_dummy))  # only P3
        self.head = Head(num_classes, kpt_shape, (width[3],))
        self.head.stride = torch.tensor([256 / h2.shape[-2]])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        h2, _, _ = self.fpn(self.net(x))  # h2 是 P3 特征
        return self.head([h2])

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class YOLO_P34(torch.nn.Module):
    """Variant 2: P3+P4 detection. Full FPN, Head on P3+P4, no P5 head."""
    def __init__(self, width, depth, num_classes, kpt_shape=(17, 3)):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)
        self.use_fpn = True

        img_dummy = torch.zeros(1, 3, 256, 256)
        h2, h4, _ = self.fpn(self.net(img_dummy))  # P3, P4
        self.head = Head(num_classes, kpt_shape, (width[3], width[4]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in [h2, h4]])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        h2, h4, _ = self.fpn(self.net(x))
        return self.head([h2, h4])

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class DarkNetP2_lite(torch.nn.Module):
    """P2 backbone without P5. Saves ~8% params vs full DarkNetP2."""
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2), C2f(width[2], width[2], depth[0], True)]
        p3 = [Conv(width[2], width[3], 3, 2), C2f(width[3], width[3], depth[1], True)]
        p4 = [Conv(width[3], width[4], 3, 2), C2f(width[4], width[4], depth[2], True)]
        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)

    def forward(self, x):
        p2 = self.p2(self.p1(x))
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        return p2, p3, p4


class DarkFPN_P2_lite(torch.nn.Module):
    """FPN for P2-P4 only: p4→p3→p2 both directions, no P5."""
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = C2f(width[3] + width[4], width[3], depth[0], False)
        self.h0 = C2f(width[2] + width[3], width[2], depth[0], False)
        self.h3 = Conv(width[2], width[2], 3, 2)
        self.h4 = C2f(width[2] + width[3], width[3], depth[0], False)
        self.h5 = Conv(width[3], width[3], 3, 2)
        self.h6 = C2f(width[3] + width[4], width[4], depth[0], False)

    def forward(self, x):
        p2, p3, p4 = x
        n3 = self.h1(torch.cat([self.up(p4), p3], 1))
        n2 = self.h0(torch.cat([self.up(n3), p2], 1))
        n3b = self.h4(torch.cat([self.h3(n2), n3], 1))
        n4b = self.h6(torch.cat([self.h5(n3b), p4], 1))
        return n3b, n4b



class DarkNet_lite(torch.nn.Module):
    """DarkNet without P5: outputs P3, P4 only."""
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2), C2f(width[2], width[2], depth[0], True)]
        p3 = [Conv(width[2], width[3], 3, 2), C2f(width[3], width[3], depth[1], True)]
        p4 = [Conv(width[3], width[4], 3, 2), C2f(width[4], width[4], depth[2], True)]
        self.p1 = torch.nn.Sequential(*p1); self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3); self.p4 = torch.nn.Sequential(*p4)
    def forward(self, x):
        p3 = self.p3(self.p2(self.p1(x)))
        return p3, self.p4(p3)

class DarkFPN_lite(torch.nn.Module):
    """FPN without P5: only P4->P3 fusion (both directions)."""
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = C2f(width[3]+width[4], width[3], depth[0], False)
        self.h2 = Conv(width[3], width[3], 3, 2)
        self.h3 = C2f(width[3]+width[4], width[4], depth[0], False)
    def forward(self, x):
        p3, p4 = x
        n3 = self.h1(torch.cat([self.up(p4), p3], 1))
        n4 = self.h3(torch.cat([self.h2(n3), p4], 1))
        return n3, n4

class YOLO_P34_lite(torch.nn.Module):
    """P34_lite: P3+P4 backbone + FPN, no P5. Middle ground."""
    def __init__(self, width, depth, num_classes, kpt_shape=(17,3)):
        super().__init__()
        self.net = DarkNet_lite(width, depth)
        self.fpn = DarkFPN_lite(width, depth)
        self.use_fpn = True
        img = torch.zeros(1,3,256,256)
        n3, n4 = self.fpn(self.net(img))
        self.head = Head(num_classes, kpt_shape, (width[3], width[4]))
        self.head.stride = torch.tensor([256/x.shape[-2] for x in [n3,n4]])
        self.stride = self.head.stride
        self.head.initialize_biases()
    def forward(self, x):
        n3, n4 = self.fpn(self.net(x))
        return self.head([n3, n4])
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m,'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class YOLO_P2_lite(torch.nn.Module):
    """P2+P3+P4 backbone + FPN, no P5. Lighter variant."""
    def __init__(self, width, depth, num_classes, kpt_shape=(17,3)):
        super().__init__()
        self.net = DarkNetP2_lite(width, depth)
        self.fpn = DarkFPN_P2_lite(width, depth)
        self.use_fpn = True
        img = torch.zeros(1,3,256,256)
        n3, n4 = self.fpn(self.net(img))
        self.head = Head(num_classes, kpt_shape, (width[3], width[4]))
        self.head.stride = torch.tensor([256/x.shape[-2] for x in [n3,n4]])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        n3, n4 = self.fpn(self.net(x))
        return self.head([n3, n4])

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self



class YOLO_P2_lite_P4(torch.nn.Module):
    """P2 backbone+FPN (no P5) with P4-only detection head."""
    def __init__(self, width, depth, num_classes, kpt_shape=(17,3)):
        super().__init__()
        self.net = DarkNetP2_lite(width, depth)
        self.fpn = DarkFPN_P2_lite(width, depth)
        self.use_fpn = True
        img = torch.zeros(1,3,256,256)
        _, n4 = self.fpn(self.net(img))
        self.head = Head(num_classes, kpt_shape, (width[4],))
        self.head.stride = torch.tensor([256/n4.shape[-2]])
        self.stride = self.head.stride
        self.head.initialize_biases()
    def forward(self, x):
        _, n4 = self.fpn(self.net(x))
        return self.head([n4])
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m,'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

import torch.nn as nn


class YOLO_P2(torch.nn.Module):
    """P2 backbone + full FPN + P3/P4 detection heads. Finest features for small objects."""
    def __init__(self, width, depth, num_classes, kpt_shape=(17, 3)):
        super().__init__()
        self.net = DarkNetP2(width, depth)
        self.fpn = DarkFPN_P2(width, depth)
        self.use_fpn = True

        img_dummy = torch.zeros(1, 3, 256, 256)
        n3, n4 = self.fpn(self.net(img_dummy))
        self.head = Head(num_classes, kpt_shape, (width[3], width[4]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in [n3, n4]])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        n3, n4 = self.fpn(self.net(x))
        return self.head([n3, n4])

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v8_n(num_classes: int = 1, kpt_shape=(17, 3)):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes, kpt_shape)


def yolo_v8_s(num_classes: int = 1, kpt_shape=(17, 3)):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes, kpt_shape)


def yolo_v8_m(num_classes: int = 1, kpt_shape=(17, 3)):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes, kpt_shape)


def yolo_v8_l(num_classes: int = 1, kpt_shape=(17, 3)):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes, kpt_shape)


def yolo_v8_x(num_classes: int = 1, kpt_shape=(17, 3)):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes, kpt_shape)


def yolo_v8_n_p3(num_classes: int = 1, kpt_shape=(17, 3)):
    """Variant 3: P3-only with full FPN."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO_P3(width, depth, num_classes, kpt_shape)


def yolo_v8_n_p34(num_classes: int = 1, kpt_shape=(17, 3)):
    """Variant 2: P3+P4 detection (no P5 head)."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO_P34(width, depth, num_classes, kpt_shape)


def yolo_v8_n_p2(num_classes: int = 1, kpt_shape=(17, 3)):
    """Variant 1: P2+P3+P4 detection for small objects."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO_P2(width, depth, num_classes, kpt_shape)


def yolo_v8_n_p2_lite(num_classes: int = 1, kpt_shape=(17, 3)):
    """Variant 1b: P2+P3+P4 backbone+FPN, no P5."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO_P2_lite(width, depth, num_classes, kpt_shape)


def yolo_v8_n_p2_lite_p4(num_classes: int = 1, kpt_shape=(17, 3)):
    """Variant 1c: P2 backbone+FPN (no P5), P4-only detection head."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO_P2_lite_P4(width, depth, num_classes, kpt_shape)


def yolo_v8_n_p34_lite(num_classes: int = 1, kpt_shape=(17, 3)):
    """P34_lite: P3+P4 backbone + FPN, no P5, no P2. ~0.96M params."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO_P34_lite(width, depth, num_classes, kpt_shape)
