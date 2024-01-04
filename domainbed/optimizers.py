import torch


def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]

    return optim_cls(params, **kwargs)


class optim_SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(optim_SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.init_mask()

    @torch.no_grad()
    def init_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = torch.zeros_like(p, requires_grad=False).cuda()

                
    @torch.no_grad()
    def update_mask(self, W, mask):
        id_list = [id(w) for w in W]
        for group in self.param_groups:
            for p in group['params']:
                if id(p) in id_list and mask[id_list.index(id(p))] is not None:
                    self.state[p]['mask'] = mask[id_list.index(id(p))].reshape(p.shape)
                    # self.state[p]['mask'].cuda()
                    self.state[p]['mask'].require_grad = False
                else:
                    self.state[p]['mask'] = torch.zeros_like(p, requires_grad=False).cuda()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        # self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    
    # @torch.no_grad()
    # def first_step(self, zero_grad=False):
    #     grad_norm = self._grad_norm()
    #     for group in self.param_groups:
    #         scale = group["rho"] / (grad_norm + 1e-12)

    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             self.state[p]["old_p"] = p.data.clone()
    #             e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
    #             p.add_(e_w)  # climb to the local maximum "w + e(w)"

    #     if zero_grad: self.zero_grad()

    # @torch.no_grad()
    # def second_step(self, zero_grad=False):
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

    #     # self.base_optimizer.step()  # do the actual "sharpness-aware" update

    #     if zero_grad: self.zero_grad()
        

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
