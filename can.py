#!coding:utf-8
from scipy.spatial.distance import cdist
import scipy.sparse.linalg as sparse
from sklearn.cluster import KMeans
import  numpy as np
import time
import faiss
from faiss import normalize_L2
import torch
import matplotlib.pyplot  as plt

class Can():
    def __init__(self,k,m,num_class):
        self.num_class = num_class;
        self.m = m;
        self.k = k;
        self.r = 0;
        self.Lambda = 0;

    def graph_init(self, X):
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(d)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        X = X.astype('float32')
        normalize_L2(X)
        gpu_index_flat.add(X)
        N = X.shape[0]
        Nidx = gpu_index_flat.ntotal
        print('There are %d samples' % Nidx);

        c = time.time()
        self.D, self.I = gpu_index_flat.search(X, self.m)
        elapsed = time.time() - c
        print('kNN Search done in %d seconds' % elapsed)

        rr = np.zeros(N);
        A = np.zeros((N, N));
        for i in range(N):
            di = np.array(self.D[i, 1:self.k + 2])

            rr[i] = 0.5 * (self.k * di[self.k] - sum(di[0:self.k]))
            id = self.I[i, 1:self.k + 2];
            c = (di[self.k] - di) / (self.k * di[self.k] - sum(di[1:self.k]) + 0.000000001);
            A[i, id] = (di[self.k] - di) / (self.k * di[self.k] - sum(di[1:self.k]) + 0.000000001);
        self.r = sum(rr) / len(rr);
        self.Lambda = self.r;
        A0 = (A + A.transpose()) / 2;
        D0 = np.eye(N) * A0.sum(axis=1);
        L0 = D0 - A0;
        plt.imshow(L0*255)
        plt.axis('off')
        plt.show()
        return L0


    def max_0(self, x):
        y = [];
        for item in x:
            if item > 0:
                y.append(item)
            else:
                y.append(0)
        return y


    def logist_0(self, x):
        flag = [];
        y = [];
        for item in x:
            if item > 0:
                flag.append(1)
                y.append(item)
            else:
                flag.append(0)
        return y, flag


    def EProjSimplex_new(self, v, k):
        # problem
        # min 1/2 ||x-v||^2
        # s.t. x>=0, 1'x = 1
        #
        ft = 1;
        n = len(v);
        v0 = v - np.mean(v) + k / n;
        vmin = np.min(v0);
        if vmin < 0:
            f = 1;
            lambada_m = 0;
            while abs(f) > 1e-10:
                v1 = v0 - lambada_m;
                [data, posidx] = self.logist_0(v1);
                g = -sum(posidx);
                f = np.sum(data) - k;
                lambada_m = lambada_m - f / g;
                ft = ft + 1;
                if ft > 100:
                    x = self.max_0(v1);
                    return x;
            return self.max_0(v1);
        else:
            return v0;


    def graph_update(self, X_s, X_f, iter):
        # X_s n*d; X_f n*d
        distf = cdist(X_f, X_f, metric='euclidean')
        N = X_s.shape[0]
        N1 = X_f.shape[0]
        if N != N1:
            print("X and F is not equ!!!");
        if iter > 5:
            d = X_s.shape[1]
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatL2(d)
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

            normalize_L2(X_s)
            gpu_index_flat.add(X_s)

            Nidx = gpu_index_flat.ntotal
            print('There are %d samples' % Nidx);

            c = time.time()
            self.D, self.I = gpu_index_flat.search(X_s, self.m)
            elapsed = time.time() - c
            print('kNN Search done in %d seconds' % elapsed)

        A = np.zeros((N, N));
        for i in range(N):
            idxa0 = self.I[i, 2:self.k + 2];
            dfi = distf[i, idxa0];
            dxi = self.D[i, idxa0];
            ad = -(dxi + self.Lambda * dfi) / (2 * self.r);
            A[i, idxa0] = self.EProjSimplex_new(ad, 1);
        A0 = (A + A.transpose()) / 2;
        D0 = np.eye(N) * A0.sum(axis=1);
        L0 = D0 - A0;
        vals, vecs = sparse.eigs(L0, k=self.num_class + 1)
        fn1 = sum(vals[0:self.num_class]);
        fn2 = sum(vals[0:self.num_class]);
        if fn1 > 0.000000001:
            self.Lambda = self.Lambda * 2;
        elif fn2 < 0.00000000001:
            self.Lambda = self.Lambda / 2;
            return 1;
        vecs[0:self.num_class]
        X = vecs.real
        rows_norm = np.linalg.norm(X, axis=1, ord=2)
        Y = (X.T / rows_norm).T
        kmeans = KMeans(n_clusters=self.num_class, random_state=1231)
        Y = kmeans.fit(Y).labels_;
        print(len(Y));
        return Y, L0, A0

    def label_progation(self):
        # label pro
        # # Create the graph
        # D = D[:, 1:] ** 3
        # I = I[:, 1:]
        # row_idx = np.arange(N)
        # row_idx_rep = np.tile(row_idx, (k, 1)).T
        # W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        # W = W + W.T
        #
        # # Normalize the graph
        # W = W - scipy.sparse.diags(W.diagonal())
        # S = W.sum(axis=1)
        # S[S == 0] = 1
        # D = np.array(1. / np.sqrt(S))
        # D = scipy.sparse.diags(D.reshape(-1))
        # Wn = D * W * D
        return 0

def dbmoon(N=100, d=2, r=10, w=2):
    N1 = 10 * N
    w2 = w / 2
    done = True
    data = np.empty(0)
    while done:
        # generate Rectangular data
        tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_y = (r + w2) * np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
        # generate double moon data ---upper
        idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
        idx = (idx.nonzero())[0]

        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
            done = False
    print(data)
    db_moon = data[0:N, :]
    print(db_moon)
    # generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon

if __name__=='__main__':
    N = 200
    d = -2
    r = 10
    w = 2
    a = 0.1
    data = dbmoon(N, d, r, w)
    plt.plot(data[0:N, 0], data[0:N, 1], 'r*', data[N:2 * N, 0], data[N:2 * N, 1], 'b*')
    plt.show()
    Graph = Can(k=15,m=50,num_class=2)
    L = Graph.graph_init(data)