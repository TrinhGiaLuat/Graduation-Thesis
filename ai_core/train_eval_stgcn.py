import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
import time
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# KIẾN TRÚC STGCN TURBO (VERSION 2.0 - FULL)
# =========================================================================

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weights)
        output = torch.matmul(adj, support)
        return output

class ST_Block(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(ST_Block, self).__init__()
        self.out_channels = out_channels
        self.temporal1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=(3, 1))
        self.spatial = GCN(out_channels, spatial_channels)
        self.temporal2 = nn.Conv2d(spatial_channels, out_channels * 2, kernel_size=(3, 1))
        self.ln = nn.LayerNorm([num_nodes, out_channels])

    def forward(self, x, adj):
        t1 = self.temporal1(x)
        t1 = (t1[:, :self.out_channels, :, :] * torch.sigmoid(t1[:, self.out_channels:, :, :]))
        s_in = t1.permute(0, 2, 3, 1)
        s_out = self.spatial(s_in.reshape(-1, s_in.size(2), s_in.size(3)), adj)
        s_out = s_out.reshape(s_in.size(0), s_in.size(1), s_in.size(2), -1).permute(0, 3, 1, 2)
        s_out = F.relu(s_out)
        t2 = self.temporal2(s_out)
        t2 = (t2[:, :self.out_channels, :, :] * torch.sigmoid(t2[:, self.out_channels:, :, :]))
        out = self.ln(t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, out_len, adj):
        super(STGCN, self).__init__()
        self.adj = adj
        self.block1 = ST_Block(in_channels, 64, 64, num_nodes)
        self.block2 = ST_Block(64, 64, 64, num_nodes)
        self.output_conv1 = nn.Conv2d(64, 128, kernel_size=(1, 1))
        self.output_conv2 = nn.Conv2d(128, out_len, kernel_size=(1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.block1(x, self.adj)
        x = self.block2(x, self.adj)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        x = torch.mean(x, dim=2)
        return x

# =========================================================================
# UTILS & HUẤN LUYỆN
# =========================================================================

def sym_adj_normalize(adj):
    adj = adj + np.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def calculate_metrics(target, prediction):
    mae = np.mean(np.abs(target - prediction))
    rmse = np.sqrt(np.mean(np.square(target - prediction)))
    mask = target > 0.5
    mape = np.mean(np.abs(target[mask] - prediction[mask]) / target[mask]) * 100 if np.sum(mask) > 0 else 0
    return mae, rmse, mape

def train_eval_stgcn():
    print("=" * 70)
    print(" [~] HUẤN LUYỆN STGCN TURBO (VERSION 2.0 - FINAL) ")
    print("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = base_dir if os.path.exists(os.path.join(base_dir, 'train.npz')) else os.path.join(base_dir, 'data', 'processed', 'pems04')

    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean_flow, std_flow = scaler['mean'][0], scaler['std'][0]
    with open(os.path.join(data_dir, 'adj_mx.pkl'), 'rb') as f:
        adj_mx = pickle.load(f)
    
    adj = torch.FloatTensor(sym_adj_normalize(adj_mx)).to(device)

    x_train = torch.FloatTensor(train_data['x'][:, :, :, 0:1])
    y_train = torch.FloatTensor(train_data['y'][:, :, :, 0])
    x_val = torch.FloatTensor(val_data['x'][:, :, :, 0:1])
    y_val = torch.FloatTensor(val_data['y'][:, :, :, 0])
    x_test = torch.FloatTensor(test_data['x'][:, :, :, 0:1])
    y_test = torch.FloatTensor(test_data['y'][:, :, :, 0])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False)

    model = STGCN(num_nodes=307, in_channels=1, out_len=12, adj=adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.L1Loss()

    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        t_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                v_loss += criterion(model(bx), by).item()
        
        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        print(f"Epoch {epoch+1:03d} | Train: {t_loss:.4f} | Val: {v_loss:.4f} | Time: {time.time()-start:.1f}s")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience: break

    torch.save(best_state, os.path.join(base_dir, 'stgcn_best_model.pt'))
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_truths = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx.to(device)).cpu().numpy()
            all_preds.append(out)
            all_truths.append(by.numpy())

    preds = np.concatenate(all_preds) * std_flow + mean_flow
    truths = np.concatenate(all_truths) * std_flow + mean_flow
    mae, rmse, mape = calculate_metrics(truths, preds)

    print("\n" + "=" * 40)
    print(" KẾT QUẢ MÔ HÌNH STGCN (TURBO V2.0):")
    print(f" - MAE:  {mae:.4f}")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print("=" * 40)

    with open(os.path.join(base_dir, 'stgcn_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\n")

if __name__ == "__main__":
    train_eval_stgcn()
