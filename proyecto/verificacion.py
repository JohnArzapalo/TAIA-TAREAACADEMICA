"""
Verificacion exhaustiva de todos los modelos
"""
import sys
import torch
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from modelos import ST_GNN

torch.manual_seed(42)
cfg  = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
crit = torch.nn.CrossEntropyLoss()

# Datos fijos reproducibles
torch.manual_seed(0)
x_train = torch.randn(32, 3, 50, 25, 2)
y_train  = torch.randint(0, 60, (32,))
x_val    = torch.randn(16, 3, 50, 25, 2)
y_val    = torch.randint(0, 60, (16,))

print('=' * 60)
print('VERIFICACION 4: El loss debe BAJAR con el entrenamiento')
print('=' * 60)

for gnn in ['gcn', 'gat', 'gin']:
    torch.manual_seed(42)
    m   = ST_GNN(in_channels=3, num_class=60, graph_cfg=cfg, gnn_type=gnn)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)

    losses = []
    for ep in range(15):
        m.train()
        opt.zero_grad()
        out  = m(x_train)
        loss = crit(out, y_train)
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 4))

    baja = losses[-1] < losses[0]
    reduccion = ((losses[0] - losses[-1]) / losses[0]) * 100
    print(f'\n  ST-{gnn.upper()}:')
    print(f'    Loss inicial : {losses[0]:.4f}')
    print(f'    Loss final   : {losses[-1]:.4f}')
    print(f'    Reduccion    : {reduccion:.1f}%')
    print(f'    Bajando?     : {baja}')
    curva = ' -> '.join(str(l) for l in losses[::3])
    print(f'    Curva (c/3ep): {curva}')
    if not baja:
        print(f'    *** ALERTA: loss no bajo en ST-{gnn.upper()} ***')

print()
print('=' * 60)
print('VERIFICACION 5: epsilon aprendido de GIN')
print('=' * 60)
torch.manual_seed(42)
m_gin = ST_GNN(in_channels=3, num_class=60, graph_cfg=cfg, gnn_type='gin')
opt   = torch.optim.SGD(m_gin.parameters(), lr=0.01, momentum=0.9)
eps_inicial = []
for block in m_gin.st_gnn_networks:
    eps_inicial.append(block.gcn.epsilon.detach().clone())

for ep in range(5):
    m_gin.train()
    opt.zero_grad()
    loss = crit(m_gin(x_train), y_train)
    loss.backward()
    opt.step()

eps_cambio = 0
for i, block in enumerate(m_gin.st_gnn_networks):
    diff = (block.gcn.epsilon - eps_inicial[i]).abs().sum().item()
    eps_cambio += diff
print(f'  Cambio total en epsilon tras 5 epocas: {eps_cambio:.6f}')
print(f'  Epsilon se actualiza: {eps_cambio > 1e-8}')

print()
print('=' * 60)
print('VERIFICACION 6: att_src/att_dst de GAT se actualizan')
print('=' * 60)
torch.manual_seed(42)
m_gat = ST_GNN(in_channels=3, num_class=60, graph_cfg=cfg, gnn_type='gat')
opt   = torch.optim.SGD(m_gat.parameters(), lr=0.01, momentum=0.9)

att_src_inicial = m_gat.st_gnn_networks[0].gcn.att_src.detach().clone()

for ep in range(5):
    m_gat.train()
    opt.zero_grad()
    loss = crit(m_gat(x_train), y_train)
    loss.backward()
    opt.step()

att_src_final = m_gat.st_gnn_networks[0].gcn.att_src.detach()
diff_att = (att_src_final - att_src_inicial).abs().sum().item()
print(f'  Cambio en att_src capa 0 tras 5 epocas: {diff_att:.6f}')
print(f'  att_src/att_dst aprenden: {diff_att > 1e-8}')

print()
print('=' * 60)
print('VERIFICACION 7: edge_importance - BUG CONFIRMADO EN GAT/GIN')
print('=' * 60)
for gnn in ['gcn', 'gat', 'gin']:
    torch.manual_seed(42)
    m = ST_GNN(in_channels=3, num_class=60, graph_cfg=cfg, gnn_type=gnn)
    loss = crit(m(x_train), y_train)
    loss.backward()
    sin_grad = [n for n, p in m.named_parameters()
                if 'edge_importance' in n and p.grad is None]
    con_grad = [n for n, p in m.named_parameters()
                if 'edge_importance' in n and p.grad is not None]
    print(f'  ST-{gnn.upper()}: edge_importance CON grad={len(con_grad)}  SIN grad={len(sin_grad)}')

print()
print('RESUMEN: si todos los checks son OK el codigo es correcto.')
