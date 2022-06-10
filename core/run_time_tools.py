# coding: utf-8

import numpy as np
from torch.utils.data import DataLoader
import torch
from core.MF import PureMF, MFDataset


def mf_with_bias(config,dataProcess):
    GPU = torch.cuda.is_available()
    device = torch.device('cuda' if GPU else "cpu")
    model = PureMF(config, dataProcess)
    model = model.to(device)

    train_dataset = MFDataset(model.data[:, 0], model.data[:, 1], model.data[:, 2],device)

    # DataLoader将一个batch_size封装成一个tensor，方便迭代
    train_iter = DataLoader(train_dataset, batch_size=1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    # model = model.float()
    for epoch in range(1000):
        model.train()
        total_loss,total_len  = 0.0, 0
        for x_u, x_i, y in train_iter:
            loss = model.cal_loss(x_u,x_i,y)

            optimizer.zero_grad()  # 清空这一批的梯度
            loss.backward()  # 回传
            optimizer.step()  # 参数更新
            total_loss += loss
            total_len += len(y)
        print('----round%2d: avg_loss: %f' % (epoch, total_loss / total_len))

    # MODEL_SAVE_PATH = config['ENV']['OUT_PUT']
    # model_save_path = os.path.join(MODEL_SAVE_PATH, "user_embedding_dim{}.pt".format(model.latent_dim))
    # torch.save(model.state_dict(), model_save_path)

    np.savetxt(config['ENV']['OUT_PUT']+'user_embedding_dim%d'%model.latent_dim,
               delimiter='\t', X=model.embedding_user.weight.detach().cpu().numpy())
    np.savetxt(config['ENV']['OUT_PUT'] + 'item_embedding_dim%d' % model.latent_dim,
               delimiter='\t', X=model.embedding_item.weight.detach().cpu().numpy())

    print('getting item embedding using PMF done with full stop count')

# to get clustering vector (users)
# params: rating_file
def clustering_vector_constructor(config,dataset):
    # cur_env = env.Env(config,dataset)
    output = config['ENV']['OUT_PUT']
    result_file_path = output +'%s_vector' % (config['USERSELECTOR']['CLUSTERING_VECTOR_TYPE'].lower())

    if config['USERSELECTOR']['CLUSTERING_VECTOR_TYPE']=='RATING':
        # 这里是userid对应的矩阵哦！
        # rating_matrix = cur_env.get_init_data()[2].toarray()[:]
        rating_matrix = dataset.sparseMatrix.toarray()[:]
        np.savetxt(X=rating_matrix, fname=result_file_path, delimiter='\t')
    elif config['USERSELECTOR']['CLUSTERING_VECTOR_TYPE']=='MF':
        np.savetxt(X=np.loadtxt(fname=output + 'user_embedding_dim%s'%(config['META']['ACTION_DIM']),delimiter='\t'),
                   fname=result_file_path, delimiter='\t')
    else:
        print('not supported clustering vector type')
        exit(0)



