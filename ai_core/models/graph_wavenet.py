import torch
import torch.nn as nn
import torch.nn.functional as F

class NConv(nn.Module):
    """
    Lớp Tích chập Đồ thị (Graph Convolution Layer) thực thi phép nhân ma trận 
    trọng số không gian nhằm trích xuất đặc trưng kề rẽ nhánh qua phương thức Einsum.
    """
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        # x tensor: (batch_size, num_features, num_nodes, seq_len)
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()

class Linear(nn.Module):
    """
    Cấu trúc Ánh xạ Tuyến tính chuyên biệt áp dụng trên bản gốc không gian Tensor 4 chiều.
    """
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class GraphConvNet(nn.Module):
    """
    Khối Tích chập Đồ thị phức hợp khai thi triển phương pháp Ma trận kề tự học (Adaptive Adjacency Matrix).
    """
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GraphConvNet, self).__init__()
        self.nconv = NConv()
        
        # c_in * (order * support_len + 1) -> Khởi điểm thông số cho cơ chế khuếch tán không gian
        c_in_k = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in_k, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphWaveNet(nn.Module):
    """
    Kiến trúc mạng Nơ-ron Đồ thị Graph WaveNet ứng dụng cho quy hoạch dự báo Không gian - Thời gian (Spatio-Temporal).
    Lõi thuật toán: Gated Temporal Convolution (Tích chập thời gian), Graph Convolutions (GCN), và Adaptive Adjacency Matrix.
    """
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        
        self.supports = supports or []
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        # Nền tảng Ma trận Kề Thích nghi (Adaptive Adjacency Matrix Formulation)
        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
            else:
                self.nodevec1 = nn.Parameter(aptinit, requires_grad=True)
                self.nodevec2 = nn.Parameter(aptinit, requires_grad=True)
            self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # Tích chập dãn cách đa lớp chồng tầng (Dilated Inception)
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # GCN định hướng kết xuất luồng dữ liệu chuẩn
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                if self.gcn_bool:
                    self.gconv.append(GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, input_data):
        # Hệ tiêu chuẩn kích thước tensor vào: (batch_size, in_dim=1, num_nodes, seq_len)
        in_len = input_data.size(3)
        if in_len < self.receptive_field:
            padding_size = self.receptive_field - in_len
            x = nn.functional.pad(input_data, (padding_size, 0, 0, 0))
        else:
            x = input_data

        x = self.start_conv(x)
        skip = 0

        # Toán tử cập nhật Ma trận kề tự sinh dựa vào ReLU & Softmax
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # Vòng duyệt đa phân (Spatio-temporal extraction operations)
        for i in range(self.blocks * self.layers):
            residual = x
            
            # Định tính cổng luồng sự kiện (Gated TCN block)
            filter_params = self.filter_convs[i](residual)
            filter_params = torch.tanh(filter_params)
            gate_params = self.gate_convs[i](residual)
            gate_params = torch.sigmoid(gate_params)
            x = filter_params * gate_params

            # Cầu nối truyền thông tin (Skip Layer Configuration)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                x = self.gconv[i](x, new_supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        # Đảo chiều để hình thành Output quy chiếu: (batch_size, pre_len=12, num_nodes)
        x = x.squeeze(3).transpose(1, 2)
        return x
