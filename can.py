#!coding:utf-8
from scipy.spatial.distance import cdist
import scipy.sparse.linalg as sparse
from sklearn.cluster import KMeans
import scipy as sp
import  numpy as np
import time
import faiss
import torch
import matplotlib.pyplot  as plt
import scipy.io as scio
import fastGNMF

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
        gpu_index_flat.add(X)
        N = X.shape[0]
        c = time.time()
        self.D, self.I = gpu_index_flat.search(X, self.m)
        elapsed = time.time() - c

        rr = np.zeros(N);
        A = np.zeros((N, N));
        for i in range(N):
            di = np.array(self.D[i, 1:self.k + 2])
            if(i==0):
                print(di)
            rr[i] = 0.5 * (self.k * di[self.k] - sum(di[0:self.k]))
            id = self.I[i, 1:self.k + 2];
            A[i, id] = (di[self.k] - di) / (self.k * di[self.k] - sum(di[1:self.k]) + 2.2204e-16);
        self.r = sum(rr) / len(rr);
        self.Lambda = self.r;
        A0 = (A + A.transpose()) / 2;
        D0 = np.eye(N) * A0.sum(axis=1);
        L0 = D0 - A0;


        return L0,A0


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
        done = False;
        distf = cdist(X_f, X_f, metric='sqeuclidean')
        N = X_s.shape[0]
        N1 = X_f.shape[0]
        if N != N1:
            print("X and F is not equ!!!");
        if iter > 5:
            d = X_s.shape[1]
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatL2(d)
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index_flat.add(X_s)
            c = time.time()
            self.D, self.I = gpu_index_flat.search(X_s, self.m)
            elapsed = time.time() - c
            print('kNN Search done in %d seconds' % elapsed)
        A = np.zeros((N, N));
        for i in range(N):
            idxa0 = self.I[i, 1:self.k + 1];
            dfi = distf[i, idxa0];
            dxi = self.D[i, 1:self.k + 1];
            ad = -(dxi + self.Lambda * dfi) / (2 * self.r);
            A[i, idxa0] = self.EProjSimplex_new(ad, 1);

        print('epoch_{}'.format(iter))

        A0 = (A + A.transpose()) / 2;
        D0 = np.eye(N) * A0.sum(axis=1);
        L0 = D0 - A0;

        vals, vecs = sparse.eigs(L0, 20, tol=0.01, which="SR")
        b = np.array(np.array(sorted(enumerate(vals.real), key=lambda x: x[1]))[:,0]).astype('int32')

        fn1 = sum(vals.real[b[0:self.num_class]]);
        fn2 = sum(vals.real[b[0:self.num_class+1]]);
        F_old = X_f;
        F = vecs.real[:,b[0:self.num_class]]
        print(vals.real[b[0:10]])
        if fn1 > 0.0001: #set value must > 0.00001 debug by yexiaocheng
            self.Lambda = self.Lambda * 2;
        elif fn2 < 0.0001: #set value must > 0.00001 debug by yexiaocheng
            self.Lambda = self.Lambda / 2; F = F_old;
            print("F = F_old")
        else:
            print('get it')
            done = True;

        kmeans = KMeans(n_clusters=self.num_class)
        Y = kmeans.fit(F).labels_;
        return done,Y, L0, A0 , F

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

        #spetcal cluster
        #
        # degreeMatrix = np.sum(A, axis=1)
        # sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
        # DWD = np.dot(np.dot(sqrtDegreeMatrix, A), sqrtDegreeMatrix)
        #
        # vals, vecs = sparse.eigs(DWD,k=2)


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
    db_moon = data[0:N, :]
    # generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon

if __name__=='__main__':

    # data = np.ascontiguousarray(scio.loadmat('dat100.mat')['ppp'],dtype='float32');
    # Graph = Can(k=15,m=20,num_class=2)
    # L, A = Graph.graph_init(data);
    # vals, vecs = sparse.eigs(L, k=2, tol=0.001, which="SR")
    # VECS = vecs.real
    #
    # for iter in range(50):
    #     done,Y, L0, A0 , F = Graph.graph_update(data,VECS,iter);
    #     plt.cla();
    #     index1 = [i for i, x in enumerate(Y) if x == 0]
    #     index2 = [i for i, x in enumerate(Y) if x == 1]
    #     plt.title('{}_epoch'.format(iter))
    #     plt.plot(data[index1, 0], data[index1, 1], 'r*')
    #     plt.plot(data[index2, 0], data[index2, 1], 'b*')
    #     plt.pause(0.1)
    #     if done:
    #         break;
    #     VECS = F;
    # plt.show()

    # groundtruth ~ to obtain the cluster labels
    X, groundtruth = fastGNMF.examples.COIL20.read_dataset(rank=10, image_num=5)

    # initialize gnmf instance with rank=10 and p=5 for p-nearest neighbors
    #  use default parameters for the rest (lambda = 0.5)
    gnmf = fastGNMF.Gnmf(X=X, rank=10, p=4)
    U, V = gnmf.factorize()

    # output a t-sne image
    fastGNMF.examples.COIL20.visualize_tsne(V, 10, groundtruth, "COIL20 test with rank=10; lambda=0.5; p=4", "test.png",
                                            tsne_perplexity=5)