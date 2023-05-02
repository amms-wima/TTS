import os
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs

from TTS.config import load_config, register_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model


import torch
import logging

from trainer.generic_utils import (
    set_partial_state_dict,
)

from trainer.io import (
    load_fsspec,
)

logger = logging.getLogger("trainer")

@dataclass
class TrainTTSArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})


class VitsTrainer(Trainer):
    def restore_model(
        self,
        config,
        restore_path: str,
        model,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler = None,
    ):
        """Restore training from an old run. It restores model, optimizer, AMP scaler and training stats.

        Args:
            config (Coqpit): Model config.
            restore_path (str): Path to the restored training run.
            model (nn.Module): Model to restored.
            optimizer (torch.optim.Optimizer): Optimizer to restore.
            scaler (torch.cuda.amp.GradScaler, optional): AMP scaler to restore. Defaults to None.

        Returns:
            Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]: [description]
        """

        def _restore_list_objs(states, obj):
            if isinstance(obj, list):
                for idx, state in enumerate(states):
                    obj[idx].load_state_dict(state)
            else:
                obj.load_state_dict(states)
            return obj

        def _get_nested_attr(src, dot_sep_attr_spec):
            attr_list = dot_sep_attr_spec.split(".")
            ret = src
            for item in attr_list:
                ret = ret.__getattr__(item)
            return ret

        def _modify_weight_tensor_size_if_required(saved_state_dict, model, is_model_module, weight_layer):
            if weight_layer in saved_state_dict:
                num_rows_checkpoint = saved_state_dict[weight_layer].shape[0]
                if (is_model_module):
                    num_rows_current_model = _get_nested_attr(model.module, weight_layer).shape[0]
                else:
                    num_rows_current_model = _get_nested_attr(model, weight_layer).shape[0]
                if num_rows_checkpoint < num_rows_current_model:
                    diff_rows = num_rows_current_model - num_rows_checkpoint
                    saved_state_dict[weight_layer] = torch.cat((saved_state_dict[weight_layer], 
                                                                        torch.zeros((diff_rows, saved_state_dict[weight_layer].shape[1]))),
                                                                        dim=0)[:num_rows_current_model, :]

        logger.info(" > Restoring from %s ...", os.path.basename(restore_path))
        checkpoint = load_fsspec(restore_path, map_location="cpu")
        # Note, neither optimizer, nor scaler were saved in the pretrained checkpoint
        try:
            logger.info(" > Restoring Optimizer...")
            optimizer = _restore_list_objs(checkpoint["optimizer"], optimizer)
        except (KeyError, TypeError, RuntimeError):
            logger.info(" > Optimizer is not compatible with the restored model.")
        if "scaler" in checkpoint and self.use_amp_scaler and checkpoint["scaler"]:
            logger.info(" > Restoring Scaler...")
            scaler = _restore_list_objs(checkpoint["scaler"], scaler)
        try:
            logger.info(" > Restoring Model...")
            _cp_dict_model = checkpoint["model"]
            _is_model_module = hasattr(model, 'module')
            _modify_weight_tensor_size_if_required(_cp_dict_model, model, _is_model_module, "text_encoder.emb.weight")
            if (model.args.use_speaker_embedding):
                _modify_weight_tensor_size_if_required(_cp_dict_model, model, _is_model_module, "emb_g.weight")
            model.load_state_dict(_cp_dict_model)
        except (KeyError, RuntimeError, ValueError):
            logger.info(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_partial_state_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        optimizer = self.restore_lr(config, self.args, model, optimizer)

        logger.info(" > Model restored from step %i", checkpoint["step"])
        restore_step = checkpoint["step"] + 1  # +1 not to immediately checkpoint if the model is restored
        restore_epoch = checkpoint["epoch"]
        torch.cuda.empty_cache()
        return model, optimizer, scaler, restore_step, restore_epoch



def main():
    """Run `tts` model training directly by a `config.json` file."""
    # init trainer args
    train_args = TrainTTSArgs()
    parser = train_args.init_argparse(arg_prefix="")

    # override trainer args from comman-line args
    args, config_overrides = parser.parse_known_args()
    train_args.parse_args(args)

    # load config.json and register
    if args.config_path or args.continue_path:
        if args.config_path:
            # init from a file
            config = load_config(args.config_path)
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        elif args.continue_path:
            # continue from a prev experiment
            config = load_config(os.path.join(args.continue_path, "config.json"))
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        else:
            # init from console args
            from TTS.config.shared_configs import BaseTrainingConfig  # pylint: disable=import-outside-toplevel

            config_base = BaseTrainingConfig()
            config_base.parse_known_args(config_overrides)
            config = register_config(config_base.model)()

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the model from config
    model = setup_model(config, train_samples + eval_samples)

    # init the trainer and 🚀
    trainer = VitsTrainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
