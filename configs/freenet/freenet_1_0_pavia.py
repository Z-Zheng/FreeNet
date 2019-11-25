config = dict(
    model=dict(
        type='FreeNet',
        params=dict(
            in_channels=103,
            num_classes=9,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewPaviaLoader',
            params=dict(
                training=True,
                num_workers=0,
                image_mat_path='./pavia/PaviaU.mat',
                gt_mat_path='./pavia/PaviaU_gt.mat',
                num_train_samples_per_class=200,
                sub_minibatch=20
            )
        ),
        test=dict(
            type='NewPaviaLoader',
            params=dict(
                training=False,
                num_workers=0,
                image_mat_path='./pavia/PaviaU.mat',
                gt_mat_path='./pavia/PaviaU_gt.mat',
                num_train_samples_per_class=200,
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
        resume_from_last=False,
    ),
    test=dict(
        draw=dict(
            image_size=(610, 340),
            palette=[
                0, 0, 0,
                192, 192, 192,
                0, 255, 1,
                0, 255, 255,
                0, 128, 1,
                255, 0, 254,
                165, 82, 40,
                129, 0, 127,
                255, 0, 0,
                255, 255, 0, ]
        )
    ),
)
