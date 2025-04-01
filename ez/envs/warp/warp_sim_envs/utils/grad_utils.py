# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections import defaultdict
from typing import Callable, Dict, List, Literal, Set, Tuple
import warp as wp
import numpy as np


# whether quaternions and transforms should be normalized before computing the finite difference Jacobian
NORMALIZE_FD_INPUTS = False

# dtypes that are not supported by autodiff
nondifferentiable_dtypes = {
    wp.bool,
    wp.int8,
    wp.uint8,
    wp.int16,
    wp.uint16,
    wp.int32,
    wp.uint32,
    wp.int64,
    wp.uint64,
}


class FontColors:
    # https://stackoverflow.com/a/287944
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def function_jacobian(
    func: Callable,
    inputs: List,
    max_outputs_per_var: int = -1,
    zero_grads: bool = True,
):
    """
    Computes the autodiff Jacobian of a function mapping from all differentiable inputs to all differentiable outputs.
    """
    tape = wp.Tape()
    with tape:
        outputs = func(*inputs)
    if isinstance(outputs, wp.array):
        outputs = [outputs]
    jac, ad_in, ad_out = tape_jacobian(tape, inputs, outputs, max_outputs_per_var=max_outputs_per_var)
    if zero_grads:
        tape.zero()
    return jac, outputs, ad_in, ad_out


def function_jacobian_fd(func: Callable, inputs: List, eps: float = 1e-4, max_fd_dims_per_var: int = 500):
    """
    Computes the Jacobian of a function mapping from all differentiable inputs to all differentiable outputs using finite differences.
    """
    diff_in = flatten_arrays(inputs)

    num_in = len(diff_in)

    assert num_in > 0, "Must specify at least one input"

    # compute numerical Jacobian
    jac_fd = None

    def eval_column(var, col_id):
        nonlocal jac_fd
        np_in = var.numpy().copy()
        np_in_original = np_in.copy()
        in_shape = np_in.shape
        np_in = np_in.flatten()
        limit = min(max_fd_dims_per_var, len(np_in)) if max_fd_dims_per_var > 0 else len(np_in)
        for j in range(limit):
            np_in[j] += eps
            var.assign(normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)))
            y1 = flatten_arrays(func(*inputs))
            if jac_fd is None:
                num_out = len(y1)
                jac_fd = np.zeros((num_out, num_in), dtype=np.float32)
            np_in[j] -= 2 * eps
            var.assign(normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)))
            y2 = flatten_arrays(func(*inputs))
            var.assign(wp.array(np_in_original, dtype=var.dtype, device=var.device))
            jac_fd[:, col_id] = (y1 - y2) / (2 * eps)
            col_id += 1
            np_in[j] = np_in_original[j]
        return col_id

    col_id = 0
    for input in inputs:
        if isinstance(input, wp.codegen.StructInstance):
            for varname, var in get_struct_vars(input).items():
                if is_differentiable(var):
                    col_id = eval_column(var, col_id)
        elif is_differentiable(input):
            col_id = eval_column(input, col_id)

    return jac_fd


def check_jacobian(
    func: Callable,
    inputs: List,
    input_names,
    output_names,
    eps: float = 1e-4,
    jacobian_name: str = "",
    max_fd_dims_per_var: int = 500,
    max_outputs_per_var: int = 500,
    atol: float = 0.1,
    rtol: float = 0.1,
    plot_jac_on_fail: bool = False,
    always_plot_jac: bool = False,
    tabulate_errors: bool = True,
):
    """
    Checks that the autodiff Jacobian of the function is correct by comparing it to the
    numerical Jacobian computed using finite differences.
    """

    jac_ad, outputs, ad_in, ad_out = function_jacobian(func, inputs, max_outputs_per_var=max_outputs_per_var)
    jac_fd = function_jacobian_fd(func, inputs, eps=eps, max_fd_dims_per_var=max_fd_dims_per_var)
    return compare_jacobians(
        jac_ad,
        jac_fd,
        inputs,
        outputs,
        input_names,
        output_names,
        jacobian_name=jacobian_name,
        max_fd_dims_per_var=max_fd_dims_per_var,
        max_outputs_per_var=max_outputs_per_var,
        ad_in=ad_in,
        ad_out=ad_out,
        atol=atol,
        rtol=rtol,
        plot_jac_on_fail=plot_jac_on_fail,
        always_plot_jac=always_plot_jac,
        tabulate_errors=tabulate_errors,
    )


def is_differentiable(x):
    # TODO add support for structs
    return isinstance(x, wp.array) and x.dtype not in nondifferentiable_dtypes


def get_struct_vars(x: wp.codegen.StructInstance):
    return {varname: getattr(x, varname) for varname, _ in x._cls.ctype._fields_}


def flatten_arrays(xs):
    # flatten arrays that are potentially differentiable
    if isinstance(xs, wp.array):
        return xs.numpy().flatten()
    arrays = []
    for x in xs:
        if isinstance(x, wp.codegen.StructInstance):
            for varname, var in get_struct_vars(x).items():
                if is_differentiable(var):
                    arrays.append(var.numpy().flatten())
        if is_differentiable(x):
            arrays.append(x.numpy().flatten())
    return np.concatenate(arrays)


def create_diff_copies(xs, require_grad=True):
    # create copies of arrays that are potentially differentiable
    diffs = []
    for x in xs:
        if isinstance(x, wp.codegen.StructInstance):
            new_struct = type(x)()
            for varname, var in get_struct_vars(x).items():
                if is_differentiable(var):
                    dvar = wp.clone(var)
                    dvar.requires_grad = require_grad
                    setattr(new_struct, varname, dvar)
                elif isinstance(var, wp.array):
                    setattr(new_struct, varname, wp.clone(var))
                else:
                    setattr(new_struct, varname, var)
            diffs.append(new_struct)
        elif is_differentiable(x):
            dx = wp.clone(x)
            dx.requires_grad = require_grad
            diffs.append(dx)
        elif isinstance(x, wp.array):
            diffs.append(wp.clone(x))
        else:
            diffs.append(x)
    return diffs


def onehot(dim, i):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


@wp.kernel
def normalize_transforms(xs: wp.array(dtype=wp.transform)):
    tid = wp.tid()
    x = xs[tid]
    xs[tid] = wp.transform(wp.transform_get_translation(x), wp.normalize(wp.transform_get_rotation(x)))


@wp.kernel
def normalize_quats(xs: wp.array(dtype=wp.quat)):
    tid = wp.tid()
    x = xs[tid]
    xs[tid] = wp.normalize(x)


def normalize_inputs(xs: wp.array):
    """
    Normalizes quaternion and transform inputs to avoid numerical issues when computing finite differences.
    Note: this operation is only performed if NORMALIZE_FD_INPUTS is set to True.
    """
    if NORMALIZE_FD_INPUTS:
        if xs.dtype == wp.transform:
            wp.launch(normalize_transforms, dim=len(xs), inputs=[xs], device=xs.device)
        elif xs.dtype == wp.quat:
            wp.launch(normalize_quats, dim=len(xs), inputs=[xs], device=xs.device)
    return xs


def get_device(xs: list):
    # retrieve best matching Warp device for a list of variables
    for x in xs:
        if isinstance(x, wp.array):
            return x.device
        elif isinstance(x, wp.codegen.StructInstance):
            for varname, var in get_struct_vars(x).items():
                if isinstance(var, wp.array):
                    return var.device
    return wp.get_preferred_device()


def tape_jacobian(
    tape: wp.Tape,
    inputs: List[wp.array],
    outputs: List[wp.array],
    max_outputs_per_var=-1,
):
    """
    Computes the Jacobian from a tape mapping from all differentiable inputs to all differentiable outputs.
    """
    if len(inputs) == 0 or len(outputs) == 0:
        return None, None, None
    diff_in = flatten_arrays(inputs)
    diff_out = flatten_arrays(outputs)

    num_in = len(diff_in)
    num_out = len(diff_out)

    assert num_in > 0, "Must specify at least one input"
    assert num_out > 0, "Must specify at least one output"

    # compute analytical Jacobian
    jac_ad = np.zeros((num_out, num_in), dtype=np.float32)

    row_id = 0

    for input_id, input in enumerate(inputs):
        if is_differentiable(input) and not input.requires_grad:
            print(
                f"{FontColors.WARNING}Error while evaluating tape Jacobian: input {input_id} (array of {input.dtype}) does not have requires_grad enabled{FontColors.ENDC}"
            )

    def eval_row(out):
        nonlocal row_id
        nonlocal tape
        nonlocal jac_ad
        np_out = out.numpy()
        out_shape = np_out.shape
        np_out = np_out.flatten()
        limit = min(max_outputs_per_var, len(np_out)) if max_outputs_per_var > 0 else len(np_out)
        for j in range(limit):
            out.grad = wp.array(onehot(len(np_out), j).reshape(out_shape), dtype=out.dtype, device=out.device)
            tape.backward()
            col_id = 0
            for input in inputs:
                # fill in Jacobian columns from input gradients
                if isinstance(input, wp.codegen.StructInstance):
                    for varname, var in get_struct_vars(input).items():
                        if is_differentiable(var):
                            grad = tape.gradients[var].numpy().flatten()
                            jac_ad[row_id, col_id : col_id + len(grad)] = grad
                            col_id += len(grad)
                elif is_differentiable(input):
                    if input not in tape.gradients:
                        col_id += len(input.numpy().flatten())
                    else:
                        grad = tape.gradients[input].numpy().flatten()
                        jac_ad[row_id, col_id : col_id + len(grad)] = grad
                        col_id += len(grad)
            tape.zero()
            row_id += 1

    for out in outputs:
        # loop over Jacobian rows, select output dimension to differentiate
        if isinstance(out, wp.codegen.StructInstance):
            for varname, var in get_struct_vars(out).items():
                if is_differentiable(var):
                    eval_row(var)
        elif is_differentiable(out):
            eval_row(out)

    return jac_ad, flatten_arrays(inputs), flatten_arrays(outputs)


def zero_vars(vars):
    # zero out the vars before executing tape operations, in case
    # there is no operation on the tape that resets the vars
    # (a loss function is often simply accumulated; array.zero_() is
    # not an operation that can be recorded on the tape)
    if isinstance(vars, (list, tuple)):
        for output in vars:
            zero_vars(output)
    elif isinstance(vars, wp.array):
        vars.zero_()
    elif isinstance(vars, wp.codegen.StructInstance):
        for varname, var in get_struct_vars(vars).items():
            zero_vars(var)


def randomize_vars(vars):
    # assign random numbers to variables
    if isinstance(vars, (list, tuple)):
        for output in vars:
            randomize_vars(output)
    elif isinstance(vars, wp.array):
        np_array = vars.numpy()
        vars.assign(np.random.rand(*np_array.shape).astype(np_array.dtype))
    elif isinstance(vars, wp.codegen.StructInstance):
        for varname, var in get_struct_vars(vars).items():
            randomize_vars(var)


def tape_jacobian_fd(
    tape: wp.Tape,
    inputs: List[wp.array],
    outputs: List[wp.array],
    eps: float = 1e-4,
    max_fd_dims_per_var: int = 500,
):
    """
    Computes the Jacobian of a Warp kernel launch mapping from all differentiable inputs to all differentiable outputs
    using finite differences.
    """

    diff_in = flatten_arrays(inputs)
    diff_out = flatten_arrays(outputs)

    num_in = len(diff_in)
    num_out = len(diff_out)

    assert num_in > 0, "Must specify at least one input"
    assert num_out > 0, "Must specify at least one output"

    zero_vars(outputs)

    def f():
        tape.forward()
        out = flatten_arrays(outputs)
        zero_vars(outputs)
        return out

    # compute numerical Jacobian
    jac_fd = np.zeros((num_out, num_in), dtype=np.float32)

    def eval_row(var, col_id):
        nonlocal jac_fd
        np_in = var.numpy().copy()
        np_in_original = np_in.copy()
        in_shape = np_in.shape
        np_in = np_in.flatten()
        limit = min(max_fd_dims_per_var, len(np_in)) if max_fd_dims_per_var > 0 else len(np_in)
        for j in range(limit):
            np_in[j] += eps
            var.assign(normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)))
            y1 = f()
            np_in[j] -= 2 * eps
            var.assign(normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)))
            y2 = f()
            var.assign(wp.array(np_in_original, dtype=var.dtype, device=var.device))
            jac_fd[:, col_id] = (y1 - y2) / (2 * eps)
            col_id += 1
            np_in[j] = np_in_original[j]
        return col_id

    col_id = 0
    for input in inputs:
        if isinstance(input, wp.codegen.StructInstance):
            for varname, var in get_struct_vars(input).items():
                if is_differentiable(var):
                    col_id = eval_row(var, col_id)
        elif is_differentiable(input):
            col_id = eval_row(input, col_id)

    return jac_fd


def kernel_jacobian(
    kernel: wp.Kernel,
    dim: int,
    inputs: List[wp.array],
    outputs: List[wp.array],
    max_outputs_per_var=-1,
    zero_grads: bool = True,
):
    """
    Computes the Jacobian of a Warp kernel launch mapping from all differentiable inputs to all differentiable outputs.
    """
    diff_inputs = create_diff_copies(inputs)
    diff_outputs = create_diff_copies(outputs)
    device = get_device(inputs + outputs)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel, dim=dim, inputs=diff_inputs, outputs=diff_outputs, device=device)
    if zero_grads:
        tape.zero()

    return tape_jacobian(tape, diff_inputs, diff_outputs, max_outputs_per_var)


def kernel_jacobian_fd(
    kernel: wp.Kernel,
    dim: int,
    inputs: List[wp.array],
    outputs: List[wp.array],
    eps: float = 1e-4,
    max_fd_dims_per_var: int = 500,
):
    """
    Computes the Jacobian of a Warp kernel launch mapping from all differentiable inputs to all differentiable outputs
    using finite differences.
    """
    assert len(outputs) > 0, "Must specify at least one output"

    diff_in = flatten_arrays(inputs)
    diff_out = flatten_arrays(outputs)

    num_in = len(diff_in)
    num_out = len(diff_out)

    diff_inputs = create_diff_copies(inputs, require_grad=False)
    device = get_device(inputs + outputs)

    def f(xs):
        # make copy of output arrays, because some kernels may modify them in-place
        diff_outputs = create_diff_copies(outputs)
        # call the kernel without taping for finite differences
        wp.launch(kernel, dim=dim, inputs=xs, outputs=diff_outputs, device=device)
        return flatten_arrays(diff_outputs)

    # compute numerical Jacobian
    jac_fd = np.zeros((num_out, num_in), dtype=np.float32)

    col_id = 0
    for input_id, input in enumerate(diff_inputs):
        if isinstance(input, wp.codegen.StructInstance):
            for varname, var in get_struct_vars(input).items():
                if is_differentiable(var):
                    np_in = var.numpy().copy()
                    np_in_original = np_in.copy()
                    np_in_original_flat = np_in_original.flatten()
                    in_shape = np_in.shape
                    np_in = np_in.flatten()
                    # choose epsilon based on input magnitude
                    eps2 = max(eps, eps * np.abs(np_in_original_flat).max())
                    # limit the number of input dimensions to evaluate
                    limit = min(max_fd_dims_per_var, len(np_in)) if max_fd_dims_per_var > 0 else len(np_in)
                    for j in range(limit):
                        x1 = np_in_original_flat[j] + eps2
                        np_in[j] = x1
                        setattr(
                            diff_inputs[input_id],
                            varname,
                            normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)),
                        )
                        y1 = f(diff_inputs)
                        x2 = np_in_original_flat[j] - eps2
                        np_in[j] = x2
                        setattr(
                            diff_inputs[input_id],
                            varname,
                            normalize_inputs(wp.array(np_in.reshape(in_shape), dtype=var.dtype, device=var.device)),
                        )
                        y2 = f(diff_inputs)
                        setattr(
                            diff_inputs[input_id], varname, wp.array(np_in_original, dtype=var.dtype, device=var.device)
                        )
                        jac_fd[:, col_id] = (y1 - y2) / (x1 - x2)
                        col_id += 1
        elif is_differentiable(input):
            np_in = input.numpy().copy()
            np_in_original = np_in.copy()
            np_in_original_flat = np_in_original.flatten()
            in_shape = np_in.shape
            np_in = np_in.flatten()
            # choose epsilon based on input magnitude
            eps2 = max(eps, eps * np.abs(np_in_original_flat).max())
            # limit the number of input dimensions to evaluate
            limit = min(max_fd_dims_per_var, len(np_in)) if max_fd_dims_per_var > 0 else len(np_in)
            for j in range(limit):
                x1 = np_in_original_flat[j] + eps2
                np_in[j] = x1
                diff_inputs[input_id] = normalize_inputs(
                    wp.array(np_in.reshape(in_shape), dtype=input.dtype, device=input.device)
                )
                y1 = f(diff_inputs)
                x2 = np_in_original_flat[j] - eps2
                np_in[j] = x2
                diff_inputs[input_id] = normalize_inputs(
                    wp.array(np_in.reshape(in_shape), dtype=input.dtype, device=input.device)
                )
                y2 = f(diff_inputs)
                diff_inputs[input_id] = wp.array(np_in_original, dtype=input.dtype, device=input.device)
                jac_fd[:, col_id] = (y1 - y2) / (x1 - x2)
                col_id += 1

    return jac_fd


def plot_matrix(
    ax, mat, vmin, vmax, input_ticks=None, input_ticks_labels=None, output_ticks=None, output_ticks_labels=None
):
    from matplotlib.colors import LogNorm

    mat = np.copy(mat)
    mat[mat == 0.0] = np.nan
    if vmin is not None and vmin < vmax and vmin > 0.0:
        ax.imshow(mat, cmap="jet", interpolation="nearest", norm=LogNorm(vmin=vmin, vmax=vmax))
    elif vmin is not None and vmin < vmax:
        ax.imshow(mat, cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)
    else:
        ax.imshow(mat, cmap="jet", interpolation="nearest")
    if input_ticks is not None and len(input_ticks) > 0:
        # grid lines should start at the beginning of each matrix entry
        ax.set_xticks([tick - 0.5 for tick in input_ticks])
        if input_ticks_labels is not None and len(input_ticks_labels) > 0:
            if len(input_ticks_labels) > 1:
                ax.set_xticklabels(
                    [f"{label} ({tick})" for label, tick in zip(input_ticks_labels, input_ticks)], rotation=90
                )
            else:
                ax.set_xticklabels(input_ticks_labels)
    if output_ticks is not None and len(output_ticks) > 0:
        # grid lines should start at the beginning of each matrix entry
        ax.set_yticks([tick - 0.5 for tick in output_ticks])
        if output_ticks_labels is not None and len(output_ticks_labels) > 0:
            if len(output_ticks_labels) > 1:
                ax.set_yticklabels([f"{label} ({tick})" for label, tick in zip(output_ticks_labels, output_ticks)])
            else:
                ax.set_yticklabels(output_ticks_labels)
    ax.grid(True)


def plot_jacobian_comparison(
    jac_ad,
    jac_fd,
    title="",
    input_ticks=None,
    input_ticks_labels=None,
    output_ticks=None,
    output_ticks_labels=None,
    highlight_xs=None,
    highlight_ys=None,
):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3)
    plt.suptitle(title, fontsize=16, fontweight="bold")

    vmin = min(np.min(jac_ad), np.min(jac_fd))
    vmax = max(np.max(jac_ad), np.max(jac_fd))
    plot_matrix(axs[0], jac_ad, vmin, vmax, input_ticks, input_ticks_labels, output_ticks, output_ticks_labels)
    axs[0].set_title("Analytical")
    plot_matrix(axs[1], jac_fd, vmin, vmax, input_ticks, input_ticks_labels, output_ticks, output_ticks_labels)
    axs[1].set_title("Finite Difference")
    diff = jac_ad - jac_fd
    plot_matrix(axs[2], diff, None, None, input_ticks, input_ticks_labels, output_ticks, output_ticks_labels)
    axs[2].set_title("Difference")
    if highlight_xs is not None and highlight_ys is not None:
        axs[2].scatter(highlight_xs, highlight_ys, marker="x", color="red")
    plt.tight_layout(h_pad=0.0, w_pad=0.0)
    # plt.tight_layout()
    plt.show()


def get_ticks(vars, labels):
    """Returns the axis ticks, labels, and sizes for a list of Warp input variables to be used for plotting matrices."""
    ticks_labels = []
    ticks = []
    lengths = {}
    i = 0
    for name, x in zip(labels, vars):
        if isinstance(x, wp.codegen.StructInstance):
            for varname, var in get_struct_vars(x).items():
                if is_differentiable(var):
                    sname = f"{name}.{varname}"
                    ticks_labels.append(sname)
                    ticks.append(i)
                    lengths[sname] = len(var.numpy().flatten())
                    i += lengths[sname]
        elif is_differentiable(x):
            ticks_labels.append(name)
            ticks.append(i)
            lengths[name] = len(x.numpy().flatten())
            i += lengths[name]
    return ticks, ticks_labels, lengths


def compare_jacobians(
    jacobian_ad,
    jacobian_fd,
    inputs,
    outputs,
    input_names,
    output_names,
    jacobian_name: str = "",
    max_fd_dims_per_var: int = 500,
    max_outputs_per_var: int = 500,
    ad_in=None,
    ad_out=None,
    atol: float = 0.1,
    rtol: float = 0.1,
    plot_jac_on_fail: bool = False,
    always_plot_jac: bool = False,
    tabulate_errors: bool = True,
):
    """
    Compare two Jacobians, one computed analytically and one computed using finite differences.
    Returns a boolean indicating whether the two Jacobians are close enough w.r.t. atol and rtol, and a dictionary of accuracy statistics.
    """
    if len(inputs) == 0:
        raise ValueError("No differentiable inputs available")
    if len(outputs) == 0:
        raise ValueError("No differentiable outputs available")

    input_ticks, input_ticks_labels, input_lengths = get_ticks(inputs, input_names)
    output_ticks, output_ticks_labels, output_lengths = get_ticks(outputs, output_names)

    def find_variable_names(idx: Tuple[int]) -> Tuple[str]:
        # idx is the row, column index in the Jacobian, need to find corresponding output, input var names
        output_label = output_ticks_labels[0]
        for i, tick in enumerate(output_ticks[1:]):
            if idx[0] >= tick:
                output_label = output_ticks_labels[i + 1]
        input_label = input_ticks_labels[0]
        for i, tick in enumerate(input_ticks[1:]):
            if idx[1] >= tick:
                input_label = input_ticks_labels[i + 1]
        return input_label, output_label

    def compute_max_abs_error(a, b):
        abs_diff = np.abs(a - b)
        max_abs_error = np.max(abs_diff)
        max_abs_error_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        return max_abs_error, max_abs_error_idx

    def compute_max_rel_error(a, b):
        denom = np.abs(a)
        absb = np.abs(b)
        denom[denom < absb] = absb[denom < absb]
        denom[denom == 0.0] = 1.0
        rel_diff = np.abs(a - b) / denom
        rel_diff[np.isnan(rel_diff)] = 0
        max_rel_error = np.max(rel_diff)
        max_rel_error_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
        return max_rel_error, max_rel_error_idx

    def compute_mean_abs_error(a, b):
        abs_diff = np.abs(a - b)
        mean_abs_error = np.mean(abs_diff)
        return mean_abs_error

    def compute_mean_rel_error(a, b):
        denom = np.maximum(np.abs(a), np.abs(b))
        denom[denom == 0.0] = 1.0
        rel_diff = np.abs(a - b) / denom
        rel_diff[np.isnan(rel_diff)] = 0
        mean_rel_error = np.mean(rel_diff)
        return mean_rel_error

    def compute_condition_number(m):
        try:
            return np.linalg.cond(m)
        except np.linalg.LinAlgError:
            return np.nan

    def colorize_error(error, tol, alt_str=None):
        if error > tol or not np.isfinite(error):
            return FontColors.FAIL + str(error) + FontColors.ENDC
        elif alt_str is None:
            return str(error)
        else:
            return alt_str

    # assert_np_equal(jac_ad, jac_fd, tol=tol)
    result = np.allclose(jacobian_ad, jacobian_fd, atol=atol, rtol=rtol)
    max_abs_error, max_abs_error_idx = compute_max_abs_error(jacobian_ad, jacobian_fd)
    labels = find_variable_names(max_abs_error_idx)
    print(
        f"Max absolute error: {colorize_error(max_abs_error, atol)} at {max_abs_error_idx} ({labels[0]} -> {labels[1]}): {jacobian_ad[max_abs_error_idx]} vs {jacobian_fd[max_abs_error_idx]}"
    )
    max_rel_error, max_rel_error_idx = compute_max_rel_error(jacobian_ad, jacobian_fd)
    labels = find_variable_names(max_rel_error_idx)
    print(
        f"Max relative error: {colorize_error(max_rel_error, rtol)} at {max_rel_error_idx} ({labels[0]} -> {labels[1]}): {jacobian_ad[max_rel_error_idx]} vs {jacobian_fd[max_rel_error_idx]}"
    )

    # compute relative condition number
    # ||J(x)|| / (||f(x)|| / ||x||)
    rel_condition_number = np.linalg.norm(jacobian_ad, ord="fro")
    if ad_in is not None and ad_out is not None:
        nfx = np.linalg.norm(ad_out, ord=2)
        if nfx > 0:
            rel_condition_number = np.linalg.norm(jacobian_ad, ord="fro") * np.linalg.norm(ad_in, ord=2) / nfx
    print(f"Relative condition number: {rel_condition_number}")

    # compute condition numbers
    cond_stat = {"total": compute_condition_number(jacobian_ad), "individual": {}}
    max_abs_error_stat = {"total": max_abs_error, "individual": {}}
    max_rel_error_stat = {"total": max_rel_error, "individual": {}}
    mean_abs_error_stat = {"total": compute_mean_abs_error(jacobian_ad, jacobian_fd), "individual": {}}
    mean_rel_error_stat = {"total": compute_mean_rel_error(jacobian_ad, jacobian_fd), "individual": {}}

    if tabulate_errors:
        headers = ["Input", "Output", "Jac Block", "Sensitivity", "Max Rel Error", "Row", "Col", "AD", "FD"]
        table = [headers]
    highlight_xs = []
    highlight_ys = []
    for input_tick, input_label in zip(input_ticks, input_ticks_labels):
        for output_tick, output_label in zip(output_ticks, output_ticks_labels):
            input_len = min(input_lengths[input_label], max_fd_dims_per_var)
            output_len = min(output_lengths[output_label], max_outputs_per_var)
            jacobian_ad_sub = jacobian_ad[output_tick : output_tick + output_len, input_tick : input_tick + input_len]
            jacobian_fd_sub = jacobian_fd[output_tick : output_tick + output_len, input_tick : input_tick + input_len]
            cond_stat["individual"][(input_label, output_label)] = compute_condition_number(jacobian_ad_sub)
            max_abs = compute_max_abs_error(jacobian_ad_sub, jacobian_fd_sub)
            max_rel = compute_max_rel_error(jacobian_ad_sub, jacobian_fd_sub)
            max_abs_error_stat["individual"][(input_label, output_label)] = max_abs[0]
            max_rel_error_stat["individual"][(input_label, output_label)] = max_rel[0]
            mean_abs_error_stat["individual"][(input_label, output_label)] = compute_mean_abs_error(
                jacobian_ad_sub, jacobian_fd_sub
            )
            mean_rel_error_stat["individual"][(input_label, output_label)] = compute_mean_rel_error(
                jacobian_ad_sub, jacobian_fd_sub
            )
            actual_idx = (max_rel[1][0] + output_tick, max_rel[1][1] + input_tick)
            if max_rel[0] > 0.0:
                highlight_xs.append(actual_idx[1])  # swap because row is vertical
                highlight_ys.append(actual_idx[0])
            if tabulate_errors:
                prefix, postfix = "", ""
                if max_abs[0] > atol or not np.isfinite(max_abs[0]):
                    prefix, postfix = FontColors.FAIL, FontColors.ENDC
                table.append(
                    [
                        prefix + input_label + postfix,
                        prefix + output_label + postfix,
                        f"[{output_tick}:{output_tick+output_len}, {input_tick}:{input_tick+input_len}]",
                        cond_stat["individual"][(input_label, output_label)],
                        max_rel[0],
                        actual_idx[0],
                        actual_idx[1],
                        f"{prefix}{jacobian_ad[actual_idx]}{postfix}",
                        f"{prefix}{jacobian_fd[actual_idx]}{postfix}",
                    ]
                )

    stats = {
        "sensitivity": {
            "Jacobian Condition Number": cond_stat,
        },
        "accuracy": {
            "Max Absolute Jacobian Error": max_abs_error_stat,
            "Max Relative Jacobian Error": max_rel_error_stat,
            "Mean Absolute Jacobian Error": mean_abs_error_stat,
            "Mean Relative Jacobian Error": mean_rel_error_stat,
        },
        "jacobian": {
            "ad": jacobian_ad,
            "fd": jacobian_fd,
        },
    }

    if tabulate_errors:
        try:
            from tabulate import tabulate

            print(tabulate(table, headers="firstrow"))
        except ImportError:
            print("Install tabulate via `pip install tabulate` to print errors")

    if always_plot_jac or not result and plot_jac_on_fail:
        plot_jacobian_comparison(
            jacobian_ad,
            jacobian_fd,
            f"{jacobian_name} Jacobian",
            input_ticks,
            input_ticks_labels,
            output_ticks,
            output_ticks_labels,
            highlight_xs,
            highlight_ys,
        )

    return result, stats


def check_kernel_jacobian(
    kernel: Callable,
    dim: Tuple[int],
    inputs: list,
    outputs: list,
    eps: float = 1e-4,
    max_fd_dims_per_var: int = 500,
    max_outputs_per_var: int = 500,
    atol: float = 0.1,
    rtol: float = 0.1,
    plot_jac_on_fail: bool = False,
    always_plot_jac: bool = False,
    tabulate_errors: bool = True,
    warn_about_missing_requires_grad: bool = True,
):
    """
    Checks whether the Jacobian of the Warp kernel is correct by comparing it to the
    numerical Jacobian computed using finite differences.
    """

    if warn_about_missing_requires_grad:
        # check that the kernel arguments have requires_grad enabled
        for input_id, input in enumerate(inputs):
            if isinstance(input, wp.codegen.StructInstance):
                for varname, var in get_struct_vars(input).items():
                    if is_differentiable(var) and not var.requires_grad:
                        print(
                            FontColors.WARNING
                            + f'Warning: input "{kernel.adj.args[input_id].label}.{varname}" is differentiable but requires_grad is False'
                            + FontColors.ENDC
                        )
            elif is_differentiable(input) and not input.requires_grad:
                print(
                    FontColors.WARNING
                    + f'Warning: input "{kernel.adj.args[input_id].label}" is differentiable but requires_grad is False'
                    + FontColors.ENDC
                )
        for output_id, output in enumerate(outputs):
            if isinstance(output, wp.codegen.StructInstance):
                for varname, var in get_struct_vars(output).items():
                    if is_differentiable(var) and not var.requires_grad:
                        print(
                            FontColors.WARNING
                            + f'Warning: output "{kernel.adj.args[output_id + len(inputs)].label}.{varname}" is differentiable but requires_grad is False'
                            + FontColors.ENDC
                        )
            elif is_differentiable(output) and not output.requires_grad:
                print(
                    FontColors.WARNING
                    + f'Warning: output "{kernel.adj.args[output_id + len(inputs)].label}" is differentiable but requires_grad is False'
                    + FontColors.ENDC
                )

    # find input/output names mapping to Jacobian indices for tick labels
    arg_names = [arg.label for arg in kernel.adj.args]
    input_names = arg_names[: len(inputs)]
    output_names = arg_names[len(inputs) :]
    jac_ad, ad_in, ad_out = kernel_jacobian(kernel, dim, inputs, outputs, max_outputs_per_var=max_outputs_per_var)
    if jac_ad is None:
        return True, "Jacobian is empty because there are either no inputs or outputs"
    jac_fd = kernel_jacobian_fd(kernel, dim, inputs, outputs, eps=eps, max_fd_dims_per_var=max_fd_dims_per_var)

    return compare_jacobians(
        jac_ad,
        jac_fd,
        inputs,
        outputs,
        ad_in=ad_in,
        ad_out=ad_out,
        jacobian_name=kernel.key,
        input_names=input_names,
        output_names=output_names,
        atol=atol,
        rtol=rtol,
        max_outputs_per_var=max_outputs_per_var,
        max_fd_dims_per_var=max_fd_dims_per_var,
        tabulate_errors=tabulate_errors,
        plot_jac_on_fail=plot_jac_on_fail,
        always_plot_jac=always_plot_jac,
    )


def check_tape_jacobians(
    tape: wp.Tape,
    inputs: list,
    outputs: list,
    input_names: list,
    output_names: list,
    eps: float = 1e-4,
    max_fd_dims_per_var: int = 500,
    max_outputs_per_var: int = 500,
    atol: float = 0.1,
    rtol: float = 0.1,
    plot_jac_on_fail: bool = True,
    tabulate_errors: bool = True,
):
    """
    Checks whether the autodiff Jacobian from the tape matches the Jacobian computed using finite differences
    over the forward pass of the operations on the tape.
    Note: this assumes that the tape contains all operations that produce the outputs from the inputs.
    """
    jac_ad, flat_ins, flat_outs = tape_jacobian(tape, inputs, outputs)
    jac_fd = tape_jacobian_fd(tape, inputs, outputs, eps=eps)
    input_ticks, input_ticks_labels = None, None
    output_ticks, output_ticks_labels = None, None
    if len(input_names) > 0:
        input_ticks, input_ticks_labels, input_lengths = get_ticks(inputs, input_names)
    if len(output_names) > 0:
        output_ticks, output_ticks_labels, output_lengths = get_ticks(outputs, output_names)

    s = f" Comparison of AD and FD Jacobians for Tape containing {len(tape.launches)} kernel launches "
    s = s.center(120, "#")
    print(FontColors.HEADER + s + FontColors.ENDC)

    result = compare_jacobians(
        jac_ad,
        jac_fd,
        inputs,
        outputs,
        ad_in=flat_ins,
        ad_out=flat_outs,
        jacobian_name="Tape",
        input_names=input_names,
        output_names=output_names,
        atol=atol,
        rtol=rtol,
        max_outputs_per_var=max_outputs_per_var,
        max_fd_dims_per_var=max_fd_dims_per_var,
        tabulate_errors=tabulate_errors,
        plot_jac_on_fail=plot_jac_on_fail,
    )

    if not plot_jac_on_fail:
        plot_jacobian_comparison(
            jac_ad, jac_fd, "Tape Jacobian", input_ticks, input_ticks_labels, output_ticks, output_ticks_labels
        )

    return result


def make_struct_of_arrays(xs):
    """Convert a nested dict of list of arrays into a struct of arrays."""

    def insert(d, target):
        for k, v in d.items():
            if isinstance(v, dict):
                insert(v, target[k])
            else:
                target[k].append(v)

    total = defaultdict(list)
    for x in xs:
        insert(x, total)
    return total


def populate_array_names(object, array_names, prefix="", postfix=""):
    if object is None or not hasattr(object, "__dict__"):
        return
    for key, value in object.__dict__.items():
        if isinstance(value, wp.array):
            array_names[value] = prefix + key + postfix
        if isinstance(value, list):
            for i, entry in enumerate(value):
                if isinstance(entry, wp.array):
                    array_names[entry] = f"{prefix}{key}[{i}]{postfix}"
                # else:
                #     populate_array_names(entry, array_names, prefix=prefix + key, postfix=f'[{i}]')
    return array_names


def check_backward_pass(
    tape: wp.Tape,
    analyze_graph=True,
    simplify_graph=True,
    render_mermaid_mmd: str = None,
    render_mermaid_html: str = None,
    render_d2: str = None,
    render_pydot_png: str = None,
    render_pydot_svg: str = None,
    render_graphviz_plot_png: str = None,
    render_graphviz_plot_svg: str = None,
    ignore_graph_scopes: bool = False,
    check_kernel_jacobians=True,
    check_input_output_jacobian=True,
    plot_jac_on_fail=False,
    jacobian_fd_eps=1e-4,
    plotting: Literal["matplotlib", "plotly", "none"] = "matplotlib",
    track_inputs: List[wp.array] = [],
    track_outputs: List[wp.array] = [],
    track_input_names: List[str] = [],
    track_output_names: List[str] = [],
    array_names: Dict[wp.array, str] = {},
    blacklist_kernels: Set[str] = set(),
    whitelist_kernels: Set[str] = set(),
    choose_longest_node_name: bool = True,
):
    """
    Runs various checks of the backward pass given the tape of recorded kernel launches.
    """

    def add_to_struct_of_arrays(d, target):
        for k, v in d.items():
            if isinstance(v, dict):
                if k not in target:
                    target[k] = defaultdict(list)
                add_to_struct_of_arrays(v, target[k])
            else:
                target[k].append(v)

    for arr, name in zip(track_inputs, track_input_names):
        array_names[arr] = name
    for arr, name in zip(track_outputs, track_output_names):
        array_names[arr] = name

    import networkx as nx

    G = nx.DiGraph()
    node_labels = {}
    kernel_launch_count = defaultdict(int)
    # array -> list of kernels that modify it
    manipulated_nodes = defaultdict(list)
    kernel_nodes = set()
    kernel_instances = dict()
    array_nodes = set()

    render_mermaid = render_mermaid_html is not None or render_mermaid_mmd is not None
    mermaid_lines = []
    d2_lines = ["direction: down"]
    d2_connections = []
    d2_prefixes = {}
    chart_indent = "\t"

    input_output_ptr = set()
    for input in track_inputs:
        input_output_ptr.add(input.ptr)
    for output in track_outputs:
        input_output_ptr.add(output.ptr)

    def add_node(G: nx.DiGraph, x: wp.array, name: str, active_scope_stack=[]):
        nonlocal node_labels
        if x in array_names:
            name = array_names[x]
        if x.ptr in node_labels:
            if x.ptr not in input_output_ptr:
                # update name unless it is an input/output array
                if choose_longest_node_name:
                    if len(name) > len(node_labels[x.ptr]):
                        node_labels[x.ptr] = name
                else:
                    node_labels[x.ptr] = name
            return
        nondifferentiable = x.dtype in nondifferentiable_dtypes
        G.add_node(
            x.ptr,
            label=f'"{name}"',
            requires_grad=x.requires_grad,
            is_kernel=False,
            nondifferentiable=nondifferentiable,
        )
        arr_id = f"arr{x.ptr}"
        type_str = str(x.dtype).split("'")[1]
        if render_mermaid:
            class_name = "array" if not x.requires_grad else "array_grad"
            mermaid_lines.append(chart_indent + f'{arr_id}(["`{name}`"]):::{class_name}')
            tooltip = (
                f"Array {name} / ptr={x.ptr}, shape={str(x.shape)}, dtype={type_str}, requires_grad={x.requires_grad}"
            )
            mermaid_lines.append(chart_indent + f'click {arr_id} callback "{tooltip}"')
        if render_d2 is not None:
            d2_lines.append(chart_indent + f'{arr_id}: "{name}"')
            d2_prefixes[arr_id] = active_scope_stack

        node_labels[x.ptr] = name

    for i, x in enumerate(track_inputs):
        if i < len(track_input_names):
            name = track_input_names[i]
        else:
            name = f"input_{i}"
        add_node(G, x, name)
    for i, x in enumerate(track_outputs):
        if i < len(track_output_names):
            name = track_output_names[i]
        else:
            name = f"output_{i}"
        add_node(G, x, name)
    # add arrays which are output of a kernel (used to simplify the graph)
    computed_nodes = set()
    for output in track_outputs:
        computed_nodes.add(output.ptr)
    active_scope_stack = []
    active_scope = None
    active_scope_id = -1
    active_scope_kernels = {}
    if not hasattr(tape, "scopes"):
        ignore_graph_scopes = True
    if not ignore_graph_scopes and len(tape.scopes) > 0:
        active_scope = tape.scopes[0]
        active_scope_id = 0
    for launch_id, launch in enumerate(tape.launches):
        if active_scope is not None:
            if launch_id == active_scope[0]:
                if active_scope[1] is None:
                    chart_indent = chart_indent[:-1]
                    if render_mermaid:
                        mermaid_lines.append(chart_indent + "end\n")
                    if render_d2 is not None:
                        d2_lines.append(chart_indent + "}\n")
                    active_scope_stack = active_scope_stack[:-1]
                else:
                    if render_mermaid:
                        mermaid_lines.append(
                            "\n" + chart_indent + f'subgraph scope{active_scope_id} ["`**{active_scope[1]}**`"]'
                        )
                    if render_d2 is not None:
                        d2_lines.append("\n" + chart_indent + f'scope{active_scope_id}: "{active_scope[1]}" {{')
                    active_scope_stack.append(f"scope{active_scope_id}")
                    chart_indent += "\t"
            # check if we are in the next scope now
            while (
                not ignore_graph_scopes
                and active_scope_id < len(tape.scopes) - 1
                and launch_id == tape.scopes[active_scope_id + 1][0]
            ):
                active_scope_id += 1
                active_scope = tape.scopes[active_scope_id]
                active_scope_kernels = {}
                if active_scope[1] is None:
                    chart_indent = chart_indent[:-1]
                    if render_mermaid:
                        mermaid_lines.append(chart_indent + "end\n")
                    if render_d2 is not None:
                        d2_lines.append(chart_indent + "}\n")
                    active_scope_stack = active_scope_stack[:-1]
                else:
                    if render_mermaid:
                        mermaid_lines.append(
                            "\n" + chart_indent + f'subgraph scope{active_scope_id} ["`**{active_scope[1]}**`"]'
                        )
                    if render_d2 is not None:
                        d2_lines.append("\n" + chart_indent + f'scope{active_scope_id}: "{active_scope[1]}" {{')
                    active_scope_stack.append(f"scope{active_scope_id}")
                    chart_indent += "\t"

        kernel = launch[0]
        dim = launch[1]
        inputs = launch[3]
        outputs = launch[4]
        if len(launch) >= 7:
            meta_data = launch[6]
        else:
            meta_data = {
                "stack_trace": "",
            }
        kernel_name = f"{kernel.key}/{kernel_launch_count[kernel.key]}"
        kernel_nodes.add(kernel_name)
        kernel_instances[kernel_name] = {
            "kernel": kernel,
            "meta_data": meta_data,
            "launch_id": launch_id,
        }
        # store kernel as node with requires_grad so that the path search works
        G.add_node(kernel_name, label=f'"{kernel_name}"', is_kernel=True, dim=dim, requires_grad=True)

        if not simplify_graph or ignore_graph_scopes or kernel.key not in active_scope_kernels:
            active_scope_kernels[kernel.key] = launch_id
            if render_mermaid:
                mermaid_lines.append(chart_indent + f'kernel{launch_id}[["`{kernel_name}`"]]:::kernel')
                tooltip = (
                    meta_data["stack_trace"][:300]
                    .replace("\n", " ")
                    .replace('"', " ")
                    .replace("'", " ")
                    .replace("[", "&#91;")
                    .replace("]", "&#93;")
                    .replace("`", "&#96;")
                    .replace(":", "&#58;")
                    .replace("\\", "&#92;")
                    .replace("/", "&#47;")
                    .replace("(", "&#40;")
                    .replace(")", "&#41;")
                    .replace(",", "")
                )
                if len(tooltip) > 0:
                    mermaid_lines.append(chart_indent + f'click kernel{launch_id} callback "{tooltip}"')
            if render_d2 is not None:
                d2_lines.append(chart_indent + f'kernel{launch_id}: "{kernel_name}"')
        node_labels[kernel_name] = kernel_name
        input_arrays = []
        for id, x in enumerate(inputs):
            name = kernel.adj.args[id].label
            if isinstance(x, wp.array):
                if x.ptr is None:
                    continue
                if not simplify_graph or x.ptr in computed_nodes or x.ptr in input_output_ptr:
                    add_node(G, x, name, active_scope_stack)
                    input_arrays.append(x.ptr)
            elif isinstance(x, wp.codegen.StructInstance):
                for varname, var in get_struct_vars(x).items():
                    if isinstance(var, wp.array):
                        if not simplify_graph or var.ptr in computed_nodes or var.ptr in input_output_ptr:
                            add_node(G, var, f"{name}.{varname}", active_scope_stack)
                            input_arrays.append(var.ptr)
        output_arrays = []
        for id, x in enumerate(outputs):
            name = kernel.adj.args[id + len(inputs)].label
            if isinstance(x, wp.array) and x.ptr is not None:
                add_node(G, x, name, active_scope_stack)
                output_arrays.append(x.ptr)
                computed_nodes.add(x.ptr)
            elif isinstance(x, wp.codegen.StructInstance):
                for varname, var in get_struct_vars(x).items():
                    if isinstance(var, wp.array):
                        add_node(G, var, f"{name}.{varname}", active_scope_stack)
                        output_arrays.append(var.ptr)
                        computed_nodes.add(var.ptr)
        if not ignore_graph_scopes and simplify_graph:
            k_id = f"kernel{active_scope_kernels[kernel.key]}"
        else:
            k_id = f"kernel{launch_id}"
        for input_x in input_arrays:
            G.add_edge(input_x, kernel_name)
            if render_mermaid:
                mermaid_lines.append(chart_indent + f"arr{input_x} --> {k_id}")
            if render_d2 is not None:
                d2_lines.append(chart_indent + f"arr{input_x} -> {k_id}")
        for output_x in output_arrays:
            # track how many kernels modify each array
            manipulated_nodes[output_x].append(kernel.key)
            G.add_edge(kernel_name, output_x)
            if render_mermaid:
                mermaid_lines.append(chart_indent + f"{k_id} --> arr{output_x}")
            if render_d2 is not None:
                d2_lines.append(chart_indent + f"{k_id} -> arr{output_x}")

        kernel_launch_count[kernel.key] += 1

    for x in node_labels:
        if x not in kernel_nodes:
            array_nodes.add(x)

    if analyze_graph:
        for x in track_inputs:
            for y in track_outputs:
                try:
                    paths = nx.all_shortest_paths(G, x.ptr, y.ptr)
                    all_differentiable = True
                    for path in paths:
                        # XXX all arrays up until the last one have to be differentiable
                        for i in path[:-1]:
                            n = G.nodes[i]
                            if not n["requires_grad"] or "nondifferentiable" in n and n["nondifferentiable"]:
                                print(
                                    FontColors.WARNING
                                    + f"Warning: nondifferentiable node on path from {node_labels[x.ptr]} to {node_labels[y.ptr]} via [{' -> '.join([node_labels[p] for p in path])}]."
                                    + FontColors.ENDC
                                )
                                print(
                                    f"Nondifferentiable array(s): [{', '.join([node_labels[p] for p in path if not G.nodes[p]['requires_grad']])}]"
                                )
                                all_differentiable = False
                                break
                            if not all_differentiable:
                                break

                    if all_differentiable:
                        many_overwrites = set(node for node in path if len(manipulated_nodes[node]) > 1)
                        if len(many_overwrites) > 0:
                            print(
                                FontColors.WARNING
                                + f"Warning: multiple kernels manipulate array(s) on path from {node_labels[x.ptr]} to {node_labels[y.ptr]}."
                                + FontColors.ENDC
                            )
                            for node in many_overwrites:
                                print(
                                    f"\tArray {node_labels[node]} is manipulated by kernels [{', '.join([kernel for kernel in manipulated_nodes[node]])}]."
                                )
                        else:
                            print(
                                FontColors.OKGREEN
                                + f"Path from {node_labels[x.ptr]} to {node_labels[y.ptr]} is differentiable."
                                + FontColors.ENDC
                            )
                except nx.NetworkXNoPath:
                    print(
                        FontColors.FAIL
                        + f"Error: there is no computation path from {node_labels[x.ptr]} to {node_labels[y.ptr]}"
                        + FontColors.ENDC
                    )

    def shorten_label(label):
        import re

        return "\n".join(re.split(r"__|_|\.", label))

    shortened_node_labels = {key: shorten_label(label) for key, label in node_labels.items()}

    if render_mermaid:
        # generate mermaid flowchart as HTML
        chart = "graph LR\n"
        mermaid_lines.append("")
        mermaid_lines.append("\tclassDef array fill:#CCCCCC")
        mermaid_lines.append("\tclassDef array_grad fill:#80E6FF")
        mermaid_lines.append("\tclassDef kernel fill:#FFB380")
        chart += "\n".join(mermaid_lines)
        if render_mermaid_mmd is not None:
            with open(render_mermaid_mmd, "w") as f:
                f.write(chart)
        if render_mermaid_html is not None:
            html = f"""<!DOCTYPE html>\n<html>
<head>
<style>
    .mermaidTooltip {{
        position: absolute;
        background: rgba(255, 255, 255, 0.95);
        padding: 5px 10px;
        border-radius: 3px;
        pointer-events: none;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        font-family: monospace;
    }}
</style>
</head>

<body>
<pre class="mermaid">
{chart}
</pre>

<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, maxEdges: 1000000, maxTextSize: 900000000, debug: true, flowchart: {{ useMaxWidth: false, htmlLabels: true }} }});
</script>
</body>

</html>
"""
            with open(render_mermaid_html, "w") as f:
                f.write(html)

                # open HTML in browser
                import webbrowser

                webbrowser.open(render_mermaid_html)

    if render_d2 is not None:
        with open(render_d2, "w") as f:
            f.write("\n".join(d2_lines))

    if render_pydot_png is not None or render_pydot_svg is not None:
        import pydot

        graph: pydot.Graph = nx.drawing.nx_pydot.to_pydot(G)

        graph.set_rankdir("TB")
        # graph.set_ranksep(2)
        # print(graph.get_nodes())
        for n in graph.get_nodes():
            n.set("label", shorten_label(n.get_attributes()["label"]))
            n.set("color", "black")
            n.set("style", "filled")
            n.set("fontname", "Arial")
            if n.get_attributes()["is_kernel"] == "True":
                n.set("shape", "rectangle")
                n.set("fillcolor", "yellow")
            else:
                n.set("shape", "ellipse")
                if n.get_attributes()["requires_grad"] == "True":
                    n.set("fillcolor", "#00B7FF4E")
                else:
                    n.set("fillcolor", "#A7A7A74E")

        if render_pydot_png is not None:
            graph.write_png(render_pydot_png)
        if render_pydot_svg is not None:
            graph.write_svg(render_pydot_svg)

    if render_graphviz_plot_png is not None or render_graphviz_plot_svg is not None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import xml.etree.ElementTree as ET

        # try:
        # pos = nx.nx_agraph.graphviz_layout(
        #     G, prog='dot', args='-Grankdir="LR"')
        pos = nx.nx_agraph.graphviz_layout(
            G,
            prog="dot",
            args='-Grankdir="TB" -Goverlap_scaling="10" -Gsize="10" -Gpad="0.0" -Gnodesep="0.5" -Gsep="10" -Granksep="3" -Gmindist="1.0"',
        )
        # pos = nx.nx_agraph.graphviz_layout(
        #     G, prog='neato', args='-Grankdir="TB" -Goverlap_scaling="10" -Gsize="10" -Gpad="0.0" -Gnodesep="0.5" -Gsep="10" -Granksep="3" -Gmindist="1.0"')
        # pos = nx.nx_pydot.pydot_layout(G)
        # pos = nx.nx_agraph.graphviz_layout(G)
        # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        # pos = nx.kamada_kawai_layout(G, scale=1e6)
        # pos = nx.spring_layout(G, seed=42, pos=pos, iterations=1, scale=100.0)
        # except:
        #     print(
        #         "Warning: could not use graphviz to layout graph. Falling back to spring layout.")
        #     print("To get better layouts, install graphviz and pygraphviz.")
        #     # pos = nx.spring_layout(G, k=100.0)
        #     pos = nx.spectral_layout(G)

        fig = plt.figure()
        fig.canvas.manager.set_window_title("Kernel launch graph")
        array_nodes = list(array_nodes)
        kernel_nodes = list(kernel_nodes)
        node_colors = []
        for x in array_nodes:
            if len(manipulated_nodes[x]) > 1:
                node_colors.append("salmon")
            elif G.nodes[x]["requires_grad"]:
                node_colors.append("lightskyblue")
            else:
                node_colors.append("lightgray")

        handles = [
            mpl.patches.Patch(color="salmon", label="multiple overwrites"),
            mpl.patches.Patch(color="lightskyblue", label="requires grad"),
            mpl.patches.Patch(color="lightgray", label="no grad"),
            mpl.patches.Patch(color="yellow", label="kernel"),
        ]
        legend = plt.legend(handles=handles, loc="upper right", prop={"size": 2})
        legend.get_frame().set_linewidth(0.2)

        node_size = 150
        font_size = 5  # 0.01
        default_draw_args = dict(alpha=0.9, edgecolors="black", linewidths=0.1, node_size=node_size)
        ax = plt.gca()
        # print(pos)
        import matplotlib.patches as patches
        from io import BytesIO

        for key, p in pos.items():
            if key not in kernel_nodes:
                continue
            w, h = node_size, node_size
            p = patches.Rectangle(
                (p[0] - w / 2, p[1] - h / 2),
                w,
                h,
                fill=True,
                clip_on=False,
                facecolor="yellow",
                edgecolor="black",
                linewidth=0.1,
            )
            p.set_gid(key)
            ax.add_artist(p)
        # first draw kernels
        # node_artists = nx.draw_networkx_nodes(G, pos, nodelist=kernel_nodes, node_color='yellow', node_shape='s', **default_draw_args)
        # for artist, (kernel_name, kernel_data) in zip(node_artists, kernel_instances.items()):
        #     kernel_instances[kernel_name]["gid"] = artist.get_gid()
        #     print(f"Kernel {kernel_name} has gid {artist.get_gid()}")
        # then draw arrays
        nx.draw_networkx_nodes(G, pos, nodelist=array_nodes, node_color=node_colors, **default_draw_args)
        nx.draw_networkx_labels(
            G,
            pos,
            labels=shortened_node_labels,
            font_size=font_size,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.0),
        )

        nx.draw_networkx_edges(
            G, pos, edgelist=G.edges(), arrows=True, edge_color="black", node_size=node_size
        )  # , width=0.1, arrowsize=2.0)

        plt.axis("off")
        if render_graphviz_plot_png is not None:
            plt.savefig(render_graphviz_plot_png, dpi=1100, bbox_inches="tight", pad_inches=0.0)

        if render_graphviz_plot_svg is not None:
            f = BytesIO()
            plt.savefig(f, format="svg", dpi=1200)
            plt.savefig("tape_graph.svg", format="svg", dpi=1200)

            tree, xmlid = ET.XMLID(f.getvalue())

            for key, kernel_data in kernel_instances.items():
                kernel_rect = tree.find(f'.//*[@id="{key}"]')
                tooltip = ET.Element("ns0:title")
                tooltip.text = kernel_data["meta_data"]["stack_trace"]
                kernel_rect.append(tooltip)

            ET.ElementTree(tree).write("tape_graph.svg")

        plt.show()

    if check_input_output_jacobian and len(track_inputs) > 0 and len(track_outputs) > 0:
        assert len(track_inputs) == len(
            track_input_names
        ), "track_inputs and track_input_names must have the same length"
        assert len(track_outputs) == len(
            track_output_names
        ), "track_outputs and track_output_names must have the same length"
        check_tape_jacobians(
            tape, track_inputs, track_outputs, track_input_names, track_output_names, plot_jac_on_fail=plot_jac_on_fail
        )

    stats = {}
    kernel_names = set()
    kernel_launch_count = defaultdict(int)
    for launch in tape.launches:
        kernel = launch[0]
        dim = launch[1]
        inputs = launch[3]
        outputs = launch[4]
        if len(whitelist_kernels) > 0 and kernel.key not in whitelist_kernels:
            continue
        if check_kernel_jacobians and kernel.key not in blacklist_kernels:
            msg = f'Checking Jacobian of kernel "{kernel.key}" (launch {kernel_launch_count[kernel.key]})...'
            print("".join(["#"] * len(msg)))
            print(FontColors.OKCYAN + msg + FontColors.ENDC)
            try:
                result, kernel_stats = check_kernel_jacobian(
                    kernel, dim, inputs, outputs, plot_jac_on_fail=plot_jac_on_fail, eps=jacobian_fd_eps
                )
                print(result)
                if isinstance(kernel_stats, str):
                    print(kernel_stats)
                else:
                    if kernel.key not in stats:
                        stats[kernel.key] = defaultdict(list)
                    add_to_struct_of_arrays(kernel_stats, stats[kernel.key])
            except Exception as e:
                print(FontColors.FAIL + f"Error while checking jacobian of kernel {kernel.key}: {e}" + FontColors.ENDC)
                raise e

        kernel_names.add(kernel.key)
        kernel_launch_count[kernel.key] += 1

    if check_kernel_jacobians and len(stats) > 0:
        # plot evolution of Jacobian statistics
        if plotting == "matplotlib":
            import itertools

            any_stat = next(iter(stats.values()))
            all_stats_names = itertools.chain.from_iterable(
                [any_stat[cat].keys() for cat in ["sensitivity", "accuracy"]]
            )
            all_stats_names = [key for key in all_stats_names if key not in ("total", "individual")]
            for kernel_name, stat in stats.items():
                import matplotlib.pyplot as plt

                num = len(all_stats_names)
                ncols = int(np.ceil(np.sqrt(num)))
                nrows = int(np.ceil(num / float(ncols)))
                fig, axes = plt.subplots(
                    ncols=ncols,
                    nrows=nrows,
                    figsize=(ncols * 5.5, nrows * 3.5),
                    squeeze=False,
                )
                fig.canvas.manager.set_window_title(kernel_name)
                plt.suptitle(kernel_name, fontsize=16, fontweight="bold")
                for dim in range(ncols * nrows):
                    ax = axes[dim // ncols, dim % ncols]
                    if dim >= num:
                        ax.axis("off")
                        continue
                kernel_stats = list(
                    itertools.chain.from_iterable([stat[cat].items() for cat in ["sensitivity", "accuracy"]])
                )
                for dim, (stat_name, cond) in enumerate(kernel_stats):
                    ax = axes[dim // ncols, dim % ncols]
                    ax.set_title(f"{stat_name}")
                    marker = "o" if len(cond["total"]) < 10 else None
                    ax.plot(cond["total"], label="total", c="k", zorder=2, marker=marker)
                    data_has_positive = False
                    for key, value in cond["individual"].items():
                        marker = "o" if len(value) < 10 else None
                        ax.plot(value, label=f"{key[0]} $\\to$ {key[1]}", zorder=1, marker=marker)
                        data_has_positive = data_has_positive or len(value) > 0 and np.any(np.array(value) > 0.0)
                    if data_has_positive:
                        ax.set_yscale("log")
                    if dim == len(kernel_stats) - 1:
                        ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.0), fancybox=True, shadow=True, ncol=2)
                    ax.grid()
                plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9, left=0.1, right=0.9, bottom=0.1)
                plt.show()
        elif plotting == "plotly":
            import dash
            from dash import Dash, dcc, html
            from dash.dependencies import Input, Output, State
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px

            external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
            app = Dash(__name__, external_stylesheets=external_stylesheets)

            kernel_names = sorted(list(kernel_names))

            colors = px.colors.qualitative.Dark24

            app.layout = html.Div(
                [
                    dcc.Store(id="appstate", data={"kernel": kernel_names[0], "mode": "jacobian"}),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=[{"label": name, "value": name} for name in kernel_names],
                                value=kernel_names[0],
                                placeholder="Select a kernel",
                                id="kernel-dropdown",
                            ),
                        ],
                        style={"width": "49%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            dcc.Tabs(
                                id="mode-selector",
                                value="jacobian",
                                children=[
                                    dcc.Tab(label="Jacobian", value="jacobian"),
                                    dcc.Tab(label="Sensitivity", value="sensitivity"),
                                    dcc.Tab(label="Accuracy", value="accuracy"),
                                ],
                            )
                        ],
                        style={"width": "49%", "display": "inline-block"},
                    ),
                    html.Div(id="view-content"),
                ]
            )

            @app.callback(
                Output("appstate", "data"),
                Input("kernel-dropdown", "value"),
                Input("mode-selector", "value"),
                State("appstate", "data"),
            )
            def update_kernel(kernel, mode, data):
                if kernel is not None:
                    data["kernel"] = kernel
                if mode is not None:
                    data["mode"] = mode
                return data

            @app.callback(Output("view-content", "children"), Input("appstate", "data"), State("appstate", "data"))
            def update_view(arg, data):
                print("selection:", data)
                selected_kernel = data["kernel"]
                selected_mode = data["mode"]
                selected_stats = stats[selected_kernel][selected_mode]
                if selected_mode == "jacobian":
                    fig = make_subplots(
                        rows=1,
                        cols=3,
                        subplot_titles=["AD Jacobian", "FD Jacobian", "Difference"],
                    )
                    fig.add_trace(px.imshow(selected_stats["ad"][0]).data[0], row=1, col=1)
                    fig.add_trace(px.imshow(selected_stats["fd"][0]).data[0], row=1, col=2)
                    fig.add_trace(px.imshow(selected_stats["ad"][0] - selected_stats["fd"][0]).data[0], row=1, col=3)
                    fig.update_layout(coloraxis=dict(colorscale="RdBu_r"))
                else:
                    selected_stat_items = selected_stats.items()
                    num = len(selected_stat_items)
                    ncols = int(np.ceil(np.sqrt(num)))
                    nrows = int(np.ceil(num / float(ncols)))
                    fig = make_subplots(
                        rows=nrows,
                        cols=ncols,
                        subplot_titles=list(selected_stats.keys()),
                        shared_xaxes=True,
                        vertical_spacing=0.07,
                        horizontal_spacing=0.05,
                    )

                    previous_labels = set()  # avoid duplicate legend entries
                    for dim, (stat_name, stat) in enumerate(selected_stat_items):
                        row = dim // ncols + 1
                        col = dim % ncols + 1
                        fig.add_trace(
                            go.Scatter(
                                y=stat["total"],
                                name="total",
                                line=dict(color="#000000"),
                                legendgroup="group0",
                                showlegend=(dim == 0),
                            ),
                            row=row,
                            col=col,
                        )
                        fig.update_yaxes(type="log", row=row, col=col)
                        for i, (key, value) in enumerate(stat["individual"].items()):
                            label = "{0} → {1}".format(key[0], key[1])
                            fig.add_trace(
                                go.Scatter(
                                    x=np.arange(len(value)),
                                    y=value,
                                    name=label,
                                    legendgroup=f"group{label}",
                                    showlegend=(label not in previous_labels),
                                    hoverlabel=dict(namelength=-1),
                                    line=dict(color=colors[i % len(colors)]),
                                ),
                                row=row,
                                col=col,
                            )
                            previous_labels.add(label)

                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                )

                return dcc.Graph(figure=fig, style={"width": "100vw", "height": "90vh"})

            # @app.callback(Output('view-content', 'children'), Input('tabs', 'value'))
            # def stat_selection(tab):
            #     try:
            #         stat_id = int(tab)
            #     except:
            #         stat_id = 0
            #     stat_name, stat = stat_items[stat_id]
            #     num = len(stat)
            #     ncols = int(np.ceil(np.sqrt(num)))
            #     nrows = int(np.ceil(num / float(ncols)))
            #     fig = make_subplots(
            #         rows=nrows, cols=ncols,
            #         subplot_titles=list(stat.keys()))

            #     previous_labels = set()  # avoid duplicate legend entries
            #     for dim, (kernel_key, cond) in enumerate(stat.items()):
            #         row = dim // ncols + 1
            #         col = dim % ncols + 1
            #         fig.add_trace(go.Scatter(
            #             y=cond["total"], name="total", line=dict(color="#000000"), legendgroup='group0', showlegend=(dim==0)), row=row, col=col)
            #         fig.update_yaxes(type="log", row=row, col=col)
            #         for key, value in cond["individual"].items():
            #             label = "{0} → {1}".format(key[0], key[1])
            #             fig.add_trace(go.Scatter(
            #                 x=np.arange(len(value)),
            #                 y=value,
            #                 name=label,
            #                 legendgroup=f'group{label}',
            #                 showlegend=(label not in previous_labels),
            #                 hoverlabel=dict(namelength=-1)), row=row, col=col)
            #             previous_labels.add(label)

            #     return dcc.Graph(figure=fig, style={'width': '100vw', 'height': '90vh'})

            app.title = "Warp backward pass statistics"
            app.run()


def check_tape_safety(
    function: Callable, inputs: list, outputs: list = None, tol: float = 1e-5, check_nans: bool = False
):
    """
    Check if all operations in the given function are recordable by a Warp tape so that the results from `tape.forward()` match the function outputs.
    """

    def flatten_outputs(outputs):
        if isinstance(outputs, (list, tuple)):
            return flatten_arrays(outputs)
        else:
            return outputs.numpy().flatten().copy()

    tape = wp.Tape()
    with tape:
        fun_outputs = function(*inputs)
    outputs = outputs or fun_outputs
    ref_output = flatten_outputs(outputs)
    # reset output arrays to a random value to ensure the tape will actually update them
    # randomize_vars(outputs)
    zero_vars(outputs)
    tape.forward(check_nans=check_nans)
    tape_output = flatten_outputs(outputs)
    print("Output from direct fn call:", ref_output)
    print("Output from tape.forward():", tape_output)
    delta = ref_output - tape_output
    err = np.max(np.abs(delta))
    if err > tol or np.isnan(err):
        print(FontColors.FAIL + "Tape output does not match function output!" + FontColors.ENDC)
        return False
    else:
        print(FontColors.OKGREEN + "Tape output matches function output." + FontColors.ENDC)
        return True


def plot_state_gradients(
    states: list,
    figure_name: str = "state_grads.html",
    blacklist_vars=set(["body_q_temp", "body_qd_temp"]),
    title: str = None,
    skip_zero_vars: bool = True,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ]

    fig = make_subplots(cols=2, subplot_titles=["Value Absolute Maximum", "Gradient Absolute Maximum"])
    if title is not None:
        fig.update_layout(title_text=title, title_x=0.5)
    absmax = defaultdict(list)
    for i, state in enumerate(states):
        for key, value in state.__dict__.items():
            if key in blacklist_vars:
                continue
            if isinstance(value, wp.array):
                if len(value) == 0 or not value.grad:
                    continue
                np_value = value.numpy()
                if skip_zero_vars and np.allclose(np_value, 0.0):
                    continue
                absmax[key].append((np.abs(np_value).max(), np.abs(value.grad.numpy()).max()))
            elif isinstance(value, list):
                for j, x in enumerate(value):
                    if isinstance(x, wp.array):
                        if len(x) == 0 or not x.grad:
                            continue
                        np_value = x.numpy()
                        if skip_zero_vars and np.allclose(np_value, 0.0):
                            continue
                        absmax[f"{key}[{j}]"].append((np.abs(np_value).max(), np.abs(x.grad.numpy()).max()))

    for i, (key, series) in enumerate(absmax.items()):
        series = np.array(series)
        val_series, grad_series = series[:, 0], series[:, 1]
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(val_series)),
                y=val_series,
                name=key,
                legendgroup=key,
                line=dict(color=color),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(grad_series)),
                y=grad_series,
                name=key,
                legendgroup=key,
                line=dict(color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(type="log")
    fig["layout"]["xaxis"]["title"] = "State"
    fig["layout"]["xaxis2"]["title"] = "State"

    # apply HTML title via JavaScript
    script = None
    if title is not None:
        script = f'document.title = "{title}";'
    fig.write_html(figure_name, auto_open=True, post_script=script)
