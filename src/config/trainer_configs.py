"""
Configurations for training various models.

Classes:
    TrainerConfig: base class for all trainer configurations.
    BidiMLPTrainerConfig: configuration of the MLP model and other settings for use in the bidirectional LSTM model.
"""
import torch
import datetime
from copy import deepcopy

class TrainerConfig:
    """
    Base class for all trainer configurations.

    Attributes:
        dataset_path_lst: list of datasets to be used in training. Should have no missing values.
        train_set_size: proportion of dataset to use for training.
        n_epochs: number of epochs to train for.
        batch_size: size of the training batch.
        lr: learning rate.
        start_idx: the starting index of the dataset to consider. If None, the dataset will start from the beginning.
        stop_idx: the final index of the dataset to consider. If None, the dataset will end at the end.
        optimizer: which optimizer to use.
        logging_dir: directory to log training information.
        logging_frequency: frequency of logging training information.
        saving_frequency: frequency of saving model checkpoints.
        lr_scheduler: whether or not to use a learning rate scheduler
        disable_tqdm: whether or not to disable the tqdm progress bar.
        resume_from_checkpoint: whether or not to resume training from a checkpoint.
        checkpoint_path: path to checkpoint to resume from.
        run_name: name of the run.
        reverse: whether or not to reverse the dataset for use in bidirectional models.
        batch_stride: by how many indices that each sequence will overlap with the previous sequence in the batch.
    """
    def __init__(
            self,
            dataset_path_lst: list,
            train_set_size: float,
            n_epochs: int,
            batch_size: int,
            lr: float = 1e-4,
            start_idx: int = None,
            stop_idx: int = None,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            logging_dir: str = "logs", 
            logging_frequency: float = 0.01, 
            saving_frequency: float = 0.01, 
            lr_scheduler: bool = False, 
            disable_tqdm: bool = False,
            resume_from_checkpoint: bool = False, 
            checkpoint_path: str = None,
            run_name: str = None,
            reverse: bool = False,
            batch_stride: int = 16
        ):
        """
        Initializes an instance of the TrainerConfig class.

        Args:
            dataset_path_lst: list of datasets to be used in training. Should have no missing values.
            train_set_size: proportion of dataset to use for training.
            n_epochs: number of epochs to train for.
            batch_size: size of the training batch.
            lr: learning rate.
            start_idx: the starting index of the dataset to consider. If None, the dataset will start from the beginning.
            stop_idx: the final index of the dataset to consider. If None, the dataset will end at the end.
            optimizer: optimizer to use.
            logging_dir: directory to log training information.
            logging_frequency: frequency of logging training information.
            saving_frequency: frequency of saving model checkpoints.
            lr_scheduler: whether or not to use a learingin rate scheduler
            disable_tqdm: whether or not to disable the tqdm progress bar.
            resume_from_checkpoint: whether or not to resume training from a checkpoint.
            checkpoint_path: path to checkpoint to resume from.
            run_name: name of the run.
            reverse: whether or not to reverse the dataset for use in bidirectional models.
            batch_stride: by how many indices that each sequence will overlap with the previous sequence in the batch.
        """
        self.dataset_path_lst = dataset_path_lst
        self.train_set_size = train_set_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.optimizer = optimizer
        self.logging_dir = logging_dir
        self.logging_frequency = logging_frequency
        self.saving_frequency = saving_frequency
        self.lr_scheduler = lr_scheduler
        self.disable_tqdm = disable_tqdm
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_path = checkpoint_path
        if run_name:
            self.run_name = run_name
        else:
            self.run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.reverse = reverse
        self.batch_stride = batch_stride

    def copy(self):
        """
        Returns a deep copy of the TrainerConfig instance.
        """
        return deepcopy(self)
    
    def __str__(self):
        return f"""TrainerConfig( dataset_path_lst: {self.dataset_path_lst}, 
            train_set_size: {self.train_set_size}, n_epochs: {self.n_epochs}, 
            batch_size: {self.batch_size}, lr: {self.lr}, start_idx: {self.start_idx}, 
            stop_idx: {self.stop_idx}, optimizer: {self.optimizer}, logging_dir: {self.logging_dir}, 
            logging_frequency: {self.logging_frequency}, saving_frequency: {self.saving_frequency}, lr_scheduler: {self.lr_scheduler}, 
            resume_from_checkpoint: {self.resume_from_checkpoint}, checkpoint_path: {self.checkpoint_path}, run_name: {self.run_name}, 
            reverse: {self.reverse}, batch_stride: {self.batch_stride})
        """

class BidiMLPTrainerConfig(TrainerConfig):
    """
    Trainer class for MLP if included in bidirectional model.

    Attributes:
        dataset_path: path to the dataset. Should have no missing values.
        train_set_size: proportion of dataset to use for training.
        n_epochs: number of epochs to train for.
        batch_size: size of the training batch.
        lr: learning rate.
        start_idx: the starting index of the dataset to consider. If None, the dataset will start from the beginning.
        stop_idx: the final index of the dataset to consider. If None, the dataset will end at the end.
        optimizer: which optimizer to use.
        logging_dir: directory to log training information.
        logging_frequency: frequency of logging training information.
        saving_frequency: frequency of saving model checkpoints.
        lr_scheduler: whether or not to use a learning rate scheduler
        resume_from_checkpoint: whether or not to resume training from a checkpoint.
        checkpoint_path: path to checkpoint to resume from.
        run_name: name of the run.
        ablation_max: maximum size of the ablation to train for the bidirectional model.
        grad_accumulation_steps: number of steps to accumulate gradients over before updating the model.
        lstm_f_cpt_file: path to the checkpoint file for the first LSTM model
        lstm_b_cpt_file: path to the checkpoint file for the second LSTM model
        """
    def __init__(
            self,
            dataset_path: str,
            train_set_size: float,
            n_epochs: int,
            batch_size: int,
            lr: float = 1e-4,
            start_idx: int = None,
            stop_idx: int = None,
            optimizer: torch.optim.Optimizer = torch.optim.Adam, 
            logging_dir: str = "logs", 
            logging_frequency: float = 0.01, 
            saving_frequency: float = 0.001, 
            lr_scheduler: bool = False, 
            resume_from_checkpoint: bool = False, 
            checkpoint_path: str = None,
            run_name: str = None,
            ablation_max: int = 50,
            grad_accumulation_steps: int = 1,
            lstm_f_cpt_file: str = None,
            lstm_b_cpt_file: str = None
        ):
        """
        Initializes an instance of the TrainerConfig class.

        Args:
            dataset_path: path to the dataset. Should have no missing values.
            train_set_size: proportion of dataset to use for training.
            n_epochs: number of epochs to train for.
            batch_size: size of the training batch.
            lr: learning rate.
            start_idx: the starting index of the dataset to consider. If None, the dataset will start from the beginning.
            stop_idx: the final index of the dataset to consider. If None, the dataset will end at the end.
            optimizer: optimizer to use.
            logging_dir: directory to log training information.
            logging_frequency: frequency of logging training information.
            saving_frequency: frequency of saving model checkpoints.
            lr_scheduler: whether or not to use a learingin rate scheduler
            resume_from_checkpoint: whether or not to resume training from a checkpoint.
            checkpoint_path: path to checkpoint to resume from.
            run_name: name of the run.
            ablation_max: maximum size of the ablation to train for the bidirectional model. 
            grad_accumulation_steps: number of steps to accumulate gradients over before updating the model.
            lstm_f_cpt_file: path to the checkpoint file for the first LSTM model
            lstm_b_cpt_file: path to the checkpoint file for the second LSTM model
        """
        super().__init__(
            dataset_path=dataset_path,
            train_set_size=train_set_size,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            start_idx=start_idx,
            stop_idx=stop_idx,
            optimizer=optimizer,
            logging_dir=logging_dir,
            logging_frequency=logging_frequency,
            saving_frequency=saving_frequency,
            lr_scheduler=lr_scheduler,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_path=checkpoint_path,
            run_name=run_name,
        )
        self.ablation_max = ablation_max
        self.grad_accumulation_steps = grad_accumulation_steps
        self.lstm_f_cpt_file = lstm_f_cpt_file
        self.lstm_b_cpt_file = lstm_b_cpt_file
    
    def __str__(self):
        return f"""
        BidiMLPTrainerConfig( n_epochs: {self.n_epochs}, lr: {self.lr}, optimizer: {self.optimizer}, 
                            lr_scheduler: {self.lr_scheduler}, ablation_max: {self.ablation_max},
                            lstm_f_cpt_file: {self.lstm_f_cpt_file}, 
                            lstm_b_cpt_file: {self.lstm_b_cpt_file})
        """