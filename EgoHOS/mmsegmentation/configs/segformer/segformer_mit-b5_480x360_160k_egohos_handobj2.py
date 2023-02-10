_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/egohos_handobj2.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    pretrained='./pretrain/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth',
    # decode_head=dict(num_classes=9), 
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=9))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=4)

train_pipeline = [dict(type='LoadAnnotations', reduce_zero_label=False)]

checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric=['mIoU', 'mFscore'], pred_eval=True, save_best='mIoU')




