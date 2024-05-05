class Config:
    voc_data_dir = './VOCdevkit/VOC2007'
    min_size = 600
    max_size = 1000
    num_workers = 1
    test_num_workers = 1

    rpn_sigma = 3.
    roi_sigma = 1.

    weight_decay = 1e-4
    lr_decay = 0.1
    lr = 0.002

    plot_every = 40

    epoch = 16

    test_num = 10000

    load_path = None

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        print(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
