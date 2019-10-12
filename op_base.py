
class op_base():
    def __init__(self,args):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.model_path = args.model_path
        self.cpu_nums = args.cpu_nums
        self.lr = args.lr


        self.train_data_num = args.train_data_num
        self.train_utils = args.train_utils
        self.summary_dir = args.summary_dir



