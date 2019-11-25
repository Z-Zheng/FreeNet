config = dict(
    model=dict(
        type='FreeNet',
        params=dict(
            in_channels=144,
            num_classes=15,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewGRSS2013Loader',
            params=dict(
                training=True,
                num_workers=0,
                image_path='./GRSS2013/2013_IEEE_GRSS_DF_Contest_CASI.tif',
                gt_path='./GRSS2013/train_roi.tif',
                sub_minibatch=20
            )
        ),
        test=dict(
            type='NewGRSS2013Loader',
            params=dict(
                training=False,
                num_workers=0,
                image_path='./GRSS2013/2013_IEEE_GRSS_DF_Contest_CASI.tif',
                gt_path='./GRSS2013/val_roi.tif',
                sub_minibatch=20
            )
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.001
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.001,
            power=0.9,
            max_iters=1000),
    ),
    train=dict(
        forward_times=1,
        num_iters=1000,
        eval_per_epoch=True,
        summary_grads=False,
        summary_weights=False,
        eval_after_train=True,
        resume_from_last=True,
    ),
    test=dict(
        draw=dict(
            image_size=(349, 1905),
            palette=[
                0, 0, 0,
                139, 67, 45,
                0, 0, 255,
                255, 100, 0,
                0, 255, 123,
                164, 75, 155,
                101, 173, 255,
                118, 254, 172,
                60, 91, 112,
                255, 255, 0,
                255, 255, 125,
                255, 0, 255,
                100, 0, 255,
                0, 172, 254,
                0, 255, 0,
                171, 175, 80
            ]
        )
    ),

)
