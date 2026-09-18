import time
from utils import *
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import data
from trainer import Trainer

class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, args, policy_net, comm, seed, *args_, **kwargs):
        self.id = id
        self.args = args
        self.policy_net = policy_net
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = None
        self.comm = comm

    def build_trainer(self):
        if self.trainer is None:
            env = data.init(self.args.env_name, self.args)
            self.trainer = Trainer(self.args, self.policy_net, env)

    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)
        self.build_trainer()

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch(epoch)
                self.trainer.optimizer.zero_grad()
                s = self.trainer.compute_grad(batch)
                merge_stat(s, stat)
                self.comm.send(stat)
            elif task == 'send_grads':
                # Keep one entry per parameter.  Omitting None gradients shifts
                # indices whenever an optional dynamic module is disabled.
                grads = [None if p.grad is None else p.grad.detach().clone()
                         for p in self.trainer.params]
                self.comm.send(grads)


class MultiProcessTrainer(object):
    def __init__(self, args, policy_net):
        self.comms = []
        self.trainer = Trainer(args, policy_net, data.init(args.env_name, args))
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, args, policy_net, comm_remote, seed=args.seed)
            worker.start()
        self.is_random = args.random

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def collect_worker_grads(self):
        # Receive fresh tensors every update.  Caching gradient pointers only
        # worked with old PyTorch zero_grad semantics and can silently reuse
        # stale gradients on newer versions.
        for comm in self.comms:
            comm.send('send_grads')
        return [comm.recv() for comm in self.comms]

    def train_batch(self, epoch):
        # run workers in parallel
        for comm in self.comms:
            comm.send(['run_batch', epoch])

        # run its own trainer
        batch, stat = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch)
        merge_stat(s, stat)

        # check if workers are finished
        for comm in self.comms:
            s = comm.recv()
            merge_stat(s, stat)

        # add gradients of workers
        worker_grads = self.collect_worker_grads()
        for param_idx, param in enumerate(self.trainer.params):
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            for gradients in worker_grads:
                worker_grad = gradients[param_idx]
                if worker_grad is not None:
                    param.grad.data.add_(worker_grad.to(param.grad.device))
            param.grad.data.div_(stat['num_steps'])

        max_grad_norm = float(getattr(self.trainer.args, 'max_grad_norm', 0.0))
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.trainer.params, max_grad_norm)

        self.trainer.optimizer.step()
        return stat

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)
