
class TaskVector():
    '''
        这段代码定义了一个名为 TaskVector 的类，用于处理任务向量相关的操作。下面是对每个方法的解释和思路流程：
        __init__ 方法
        该方法用于初始化任务向量对象，可以通过预训练的检查点和微调后的检查点来创建任务向量，或者直接传入任务向量的状态字典。
        如果传入了任务向量状态字典，则直接使用该状态字典作为任务向量；否则，从预训练和微调的状态字典中计算任务向量。

        __add__ 方法
        该方法用于将两个任务向量相加，返回一个新的任务向量对象。
        遍历每个键，将两个任务向量对应键的值相加，得到新的任务向量。

        __radd__ 方法
        该方法定义了反向相加操作，当另一个对象为整数或空时，返回本身；否则调用__add__ 方法实现加法操作。

        __neg__ 方法
        该方法用于对任务向量进行取负操作，返回一个新的任务向量对象。
        遍历每个键，将任务向量的值取负，得到新的任务向量。

        weightmerging 方法
        该方法用于对多个任务向量进行加权合并，返回一个新的任务向量对象。
        遍历每个键，对多个任务向量对应键的加权和进行计算，得到新的任务向量。

        apply_to 方法
        该方法用于将任务向量应用到预训练模型中，返回应用了任务向量的新模型。
        遍历每个键，对预训练模型的状态字典进行相应的更新，得到新的模型。
        整体上，这个 TaskVector 类拥有一系列对任务向量进行操作的方法，能够方便地进行任务向量的组合、应用和转换等操作
        
    '''
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """

        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                print('TaskVector:' + finetuned_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model





class TaskVector_Ties():
    '''
    这个类用于处理任务向量相关的操作，提供了不同的方法来处理和组合任务向量。
    '''
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, task_vector_dict=None):
        if task_vector_dict:
            self.task_vector = task_vector_dict
        else:
            self.task_vector = self.calculate_task_vector(pretrained_checkpoint, finetuned_checkpoint)

    def calculate_task_vector(self, pretrained_checkpoint, finetuned_checkpoint):
        # 这个方法需要将预训练和微调的检查点相结合计算出任务向量
        # 返回计算后的任务向量
        # 由于缺失的具体实现细节，这里只是一个示意性的空方法
        return {}

    def __add__(self, other):
        if isinstance(other, TaskVector):
            new_task_vector = TaskVector(task_vector_dict={})
            for key in self.task_vector:
                new_task_vector.task_vector[key] = self.task_vector[key] + other.task_vector.get(key, 0)
            return new_task_vector
        else:
            raise ValueError("Can only add another TaskVector object")

    def __radd__(self, other):
        if other == 0:
            # 这允许使用sum()函数在一个空的TaskVector基础上添加TaskVector对象
            return self
        else:
            return self.__add__(other)

    def __neg__(self):
        new_task_vector = TaskVector(task_vector_dict={})
        for key in self.task_vector:
            new_task_vector.task_vector[key] = -self.task_vector[key]
        return new_task_vector

    def weightmerging(self, other_task_vectors, weights):
        new_task_vector = TaskVector(task_vector_dict={})
        for key in self.task_vector:
            weighted_sum = self.task_vector[key] * weights[0]
            for i, other_task_vector in enumerate(other_task_vectors):
                weighted_sum += other_task_vector.task_vector.get(key, 0) * weights[i+1]
            new_task_vector.task_vector[key] = weighted_sum
        return new_task_vector

    def apply_to(self, pretrained_model):
        # 这个方法需要将任务向量应用到预训练模型中
        # 返回应用了任务向量的新模型
        # 由于缺失的具体实现细节，这里只是一个示意性的空方法
        return pretrained_model

    def ties_merging(
        self,
        reset_thresh=None,
        merge_func=""
    ):
        print("RESOLVING REDUNDANCY")
        all_checks = self.task_vector.clone()
        updated_checks, *_ = topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
        print("RESOLVING SIGN")
        final_signs = resolve_sign(updated_checks)
        assert final_signs is not None

        print(f"Disjoint AGGREGATION: {merge_func}")
        merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

        return merged_tv

# 假定的外部函数，这些实现细节还需要定义
def topk_values_mask(all_checks, K, return_mask):
    # 实现选择前K%大的参数，mask剩余的参数
    pass

def resolve_sign(updated_checks):
    # 实现解析检查值的符号
    pass

def disjoint_merge(updated_checks, merge_func, final_signs):
    # 实现根据merge_func和符号合并检查值
    pass

def topk_values_mask(M, K=0.7, return_mask=False):
    '''
    这段代码的主要思路是找出给定张量 M 中每行前 K%（或 K/100）个绝对值最大的元素，
    并将其它元素位置上的数值置为0，从而得到一个 mask 张量，
    该张量在每行中前 K% 的元素位置上为 True，其余位置为 False。
    '''
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_sign(Tensor):
    '''
    计算一个张量 `Tensor` 在每一列上的符号，
    即根据每一列的元素值的正负号计算该列的符号，并将计算所得的每列符号放到一个张量中返回。
    '''
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

def resolve_zero_signs(sign_to_mult, method="majority"):
    '''
    修正符号的计算结果，避免出现符号均为 0 的情况。
    '''
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult





def disjoint_merge(Tensor, merge_func, sign_to_mult):
    '''
    根据提供的合并函数和给定的符号信息对一个张量进行聚合操作，并返回聚合结果。
    '''
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs

