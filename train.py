from simplecv import dp_train as train
import torch
from simplecv.util.logger import eval_progress, speed
import time
from module import freenet
from simplecv.util import metric
from data import dataloader


def fcn_evaluate_fn(self, test_dataloader, config):
    if self.checkpoint.global_step < 0:
        return
    self._model.eval()
    total_time = 0.
    with torch.no_grad():
        for idx, (im, mask, w) in enumerate(test_dataloader):
            start = time.time()
            y_pred = self._model(im).squeeze()
            torch.cuda.synchronize()
            time_cost = round(time.time() - start, 3)
            y_pred = y_pred.argmax(dim=0).cpu() + 1
            w.unsqueeze_(dim=0)

            w = w.byte()
            mask = torch.masked_select(mask.view(-1), w.view(-1))
            y_pred = torch.masked_select(y_pred.view(-1), w.view(-1))

            oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
            aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 self._model.module.config.num_classes,
                                                                 return_accuracys=True)
            kappa = metric.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self._model.module.config.num_classes)
            total_time += time_cost
            speed(self._logger, time_cost, 'im')

            eval_progress(self._logger, idx + 1, len(test_dataloader))

    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batched im (avg)')

    metric_dict = {
        'OA': oa.item(),
        'AA': aa.item(),
        'Kappa': kappa.item()
    }
    for i, acc in enumerate(acc_per_class):
        metric_dict['acc_{}'.format(i + 1)] = acc.item()
    self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint.global_step)


def register_evaluate_fn(launcher):
    launcher.override_evaluate(fcn_evaluate_fn)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = train.parser.parse_args()
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    train.run(config_path=args.config_path,
              model_dir=args.model_dir,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[register_evaluate_fn],
              opts=args.opts)
