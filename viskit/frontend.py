
import sys
sys.path.append('.')
import matplotlib
import os

matplotlib.use('Agg')
import flask  # import Flask, render_template, send_from_directory
from rllab.misc.ext import flatten
from rllab.viskit import core
from rllab.misc import ext
import sys
import argparse
import json
import numpy as np
# import threading, webbrowser
import plotly.offline as po
import plotly.graph_objs as go

# Taken from plot.ly
color_defaults = [
    '#1163D3',  # blue
    '#FEC20C',  # orange
    '#E66100',  # red
    '#5D399B',  # purple
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]



def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


import itertools

app = flask.Flask(__name__, static_url_path='/static')

exps_data = None
plottable_keys = None
distinct_params = None

@app.route('/js/<path:path>')
def send_js(path):
    return flask.send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return flask.send_from_directory('css', path)


def make_plot(plot_list, use_median=False, plot_width=None, plot_height=None, title=None):
    data = []
    p25, p50, p75 = [], [], []
    for idx, plt in enumerate(plot_list):
        color = color_defaults[idx % len(color_defaults)]
        if use_median:
            p25.append(np.mean(plt.percentile25))
            p50.append(np.mean(plt.percentile50))
            p75.append(np.mean(plt.percentile75))
            x = list(range(len(plt.percentile50)))
            y = list(plt.percentile50)
            y_upper = list(plt.percentile75)
            y_lower = list(plt.percentile25)
        else:
            x = list(range(len(plt.means)))
            y = list(plt.means)
            y_upper = list(plt.means + plt.stds)
            y_lower = list(plt.means - plt.stds)

        data.append(go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='tozerox',
            fillcolor=core.hex_to_rgb(color, 0.2),
            line=go.Line(color='rgba(255,255,255,0)'), #'transparent'),
            showlegend=False,
            legendgroup=plt.legend,
            hoverinfo='none'
        ))
        data.append(go.Scatter(
            x=x,
            y=y,
            name=plt.legend,
            legendgroup=plt.legend,
            line=dict(color=core.hex_to_rgb(color)),
        ))
    p25str = '['
    p50str = '['
    p75str = '['
    for p25e, p50e, p75e in zip(p25, p50, p75):
        p25str += (str(p25e) + ',')
        p50str += (str(p50e) + ',')
        p75str += (str(p75e) + ',')
    p25str += ']'
    p50str += ']'
    p75str += ']'
    print(p25str)
    print(p50str)
    print(p75str)

    layout = go.Layout(
        legend=dict(
            x=1,
            y=1,
            # xanchor="left",
            # yanchor="bottom",
        ),
        width=plot_width,
        height=plot_height,
        title=title,
    )
    fig = go.Figure(data=data, layout=layout)
    fig_div = po.plot(fig, output_type='div', include_plotlyjs=False)
    if "footnote" in plot_list[0]:
        footnote = "<br />".join([
            r"<span><b>%s</b></span>: <span>%s</span>" % (plt.legend, plt.footnote)
            for plt in plot_list
        ])
        return r"%s<div>%s</div>" % (fig_div, footnote)
    else:
        return fig_div


def make_plot_eps(plot_list, use_median=False, counter=0, plot_key=""):
    import matplotlib.pyplot as _plt

    new_plot_list = []
    new_plot_list_end = []

    red_select = "0.125"
    red_idx = None

    for idx,pl in enumerate(plot_list):
        try:
            float(pl.legend)
            if pl.legend == red_select:
                red_idx = idx

            pl.legend = '$\sigma_m = $' + pl.legend

        except:
            pass


        if 'VASE' in pl.legend:
            new_plot_list_end.append(pl)
        else:
            new_plot_list.append(pl)

    plot_list = new_plot_list + new_plot_list_end


    f, ax = _plt.subplots(figsize=(8, 5))
    for idx, plt in enumerate(plot_list):

        if red_idx is not None:
            if idx == red_idx:
                color = color_defaults[3]
            elif idx == 3:
                color = color_defaults[red_idx % len(color_defaults)]
            else:
                color = color_defaults[idx % len(color_defaults)]
        else:
            color = color_defaults[idx % len(color_defaults)]


        if use_median:
            x = list(range(len(plt.percentile50)))
            y = list(plt.percentile50)
            y_upper = list(plt.percentile75)
            y_lower = list(plt.percentile25)
        else:
            x = list(range(len(plt.means)))
            y = list(plt.means)
            y_upper = list(plt.means + plt.stds)
            y_lower = list(plt.means - plt.stds)

        ax.fill_between(
            x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3)

        ax.plot(x, y, color=color, label=plt.legend, linewidth=2.0)
        ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        loc = 'upper left'

        leg = ax.legend(loc=loc, prop={'size': 12}, ncol=1)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)

        def y_fmt(x, y):
            return str(int(np.round(x / 1000.0))) + 'K'

        if plot_key == "AverageReturn":
            plotStr = "Average cumulative extrinsic reward"
        elif plot_key == "AverageDiscountedReturn":
            plotStr = "Average discounted extrinsic reward"
        else:
            plotStr = plot_key


        ax.set_ylabel(plotStr)
        ax.set_xlabel('Iterations')

        import matplotlib.ticker as tick

        if len(args.data_paths)==1:
            eps_filename = args.data_paths[0].split('/')[-1]
        else:
            eps_filename = "tmp"

        if not os.path.exists('figures'):
            os.mkdir('figures')

        _plt.savefig("figures/" + eps_filename + "_" + str(counter) + '.pdf', bbox_inches='tight')


def summary_name(exp, selector=None):
    return exp.params["exp_name"]


def check_nan(exp):
    return all(not np.any(np.isnan(vals)) for vals in list(exp.progress.values()))


def get_plot_instruction(plot_key, split_key=None, group_key=None, filters=None, use_median=False,
                         only_show_best=False, only_show_best_final=False, gen_eps=False,
                         only_show_best_sofar=False, clip_plot_value=None, plot_width=None,
                         plot_height=None, filter_nan=False, smooth_curve=False, custom_filter=None,
                         legend_post_processor=None, normalize_error=False, custom_series_splitter=None):
    print(plot_key, split_key, group_key, filters)
    if filter_nan:
        nonnan_exps_data = list(filter(check_nan, exps_data))
        selector = core.Selector(nonnan_exps_data)
    else:
        selector = core.Selector(exps_data)
    if legend_post_processor is None:
        legend_post_processor = lambda x: x
    if filters is None:
        filters = dict()
    for k, v in filters.items():
        selector = selector.where(k, str(v))
    if custom_filter is not None:
        selector = selector.custom_filter(custom_filter)
    # print selector._filters

    if split_key is not None:
        vs = [vs for k, vs in distinct_params if k == split_key][0]
        split_selectors = [selector.where(split_key, v) for v in vs]
        split_legends = list(map(str, vs))
    else:
        split_selectors = [selector]
        split_legends = ["Plot"]
    plots = []
    counter = 1
    for split_selector, split_legend in zip(split_selectors, split_legends):
        if custom_series_splitter is not None:
            exps = split_selector.extract()
            splitted_dict = dict()
            for exp in exps:
                key = custom_series_splitter(exp)
                if key not in splitted_dict:
                    splitted_dict[key] = list()
                splitted_dict[key].append(exp)
            splitted = list(splitted_dict.items())
            group_selectors = [core.Selector(list(x[1])) for x in splitted]
            group_legends = [x[0] for x in splitted]
        else:
            if group_key and group_key is not "exp_name":
                vs = [vs for k, vs in distinct_params if k == group_key][0]
                group_selectors = [split_selector.where(group_key, v) for v in vs]
                group_legends = [str(x) for x in vs]
            else:
                group_key = "exp_name"
                vs = sorted([x.params["exp_name"] for x in split_selector.extract()])
                group_selectors = [split_selector.where(group_key, v) for v in vs]
                group_legends = [summary_name(x.extract()[0], split_selector) for x in group_selectors]

        #Place VASE plots at the end
        group_selectors_pre = []
        group_selectors_post = []
        group_legends_pre = []
        group_legends_post = []



        for i,group_legend in enumerate(group_legends):
          if group_legend == 'sandbox.vase.algos.trpo_expl.TRPO':
              group_selectors_post.append(group_selectors[i])
              group_legends_post.append(group_legend)
          else:
              group_selectors_pre.append(group_selectors[i])
              group_legends_pre.append(group_legend)

        group_selectors = group_selectors_pre + group_selectors_post
        group_legends = group_legends_pre + group_legends_post

        to_plot = []
        for group_selector, group_legend in zip(group_selectors, group_legends):

            if group_legend == 'rllab.algos.trpo.TRPO':
                legendStr = 'TRPO'
            elif group_legend == 'sandbox.jachiam.algos.plus.trpo_plus.TRPO':
                legendStr = 'TRPO+NLL'
            elif group_legend == 'sandbox.vase.algos.trpo_expl.TRPO':
                legendStr = 'TRPO+VASE'
            elif group_legend == 'sandbox.vime.algos.trpo_expl.TRPO':
                legendStr = 'TRPO+VIME'
            else:
                legendStr = group_legend

            filtered_data = group_selector.extract()
            if len(filtered_data) > 0:

                if only_show_best or only_show_best_final or only_show_best_sofar:
                    # Group by seed and sort.
                    # -----------------------
                    filtered_params = core.extract_distinct_params(filtered_data, l=0)
                    filtered_params2 = [p[1] for p in filtered_params]
                    filtered_params_k = [p[0] for p in filtered_params]
                    product_space = list(itertools.product(
                        *filtered_params2
                    ))
                    data_best_regret = None
                    best_regret = -np.inf
                    kv_string_best_regret = None
                    for idx, params in enumerate(product_space):
                        selector = core.Selector(exps_data)
                        for k, v in zip(filtered_params_k, params):
                            selector = selector.where(k, str(v))
                        data = selector.extract()
                        if len(data) > 0:
                            progresses = [
                                exp.progress.get(plot_key, np.array([np.nan])) for exp in data
                            ]

                            sizes = list(map(len, progresses))
                            max_size = max(sizes)
                            progresses = [
                                np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]

                            if only_show_best_final:
                                progresses = np.asarray(progresses)[:, -1]
                            if only_show_best_sofar:
                                progresses =np.max(np.asarray(progresses), axis=1)
                            if use_median:
                                medians = np.nanmedian(progresses, axis=0)
                                regret = np.mean(medians)
                            else:
                                means = np.nanmean(progresses, axis=0)
                                regret = np.mean(means)
                            distinct_params_k = [p[0] for p in distinct_params]
                            distinct_params_v = [
                                v for k, v in zip(filtered_params_k, params) if k in distinct_params_k]
                            distinct_params_kv = [
                                (k, v) for k, v in zip(distinct_params_k, distinct_params_v)]
                            distinct_params_kv_string = str(
                                distinct_params_kv).replace('), ', ')\t')
                            print(
                                '{}\t{}\t{}'.format(regret, len(progresses), distinct_params_kv_string))
                            if regret > best_regret:
                                best_regret = regret
                                best_progress = progresses
                                data_best_regret = data
                                kv_string_best_regret = distinct_params_kv_string

                    print(group_selector._filters)
                    print('best regret: {}'.format(best_regret))
                    # -----------------------
                    if best_regret != -np.inf:
                        progresses = [
                            exp.progress.get(plot_key, np.array([np.nan])) for exp in data_best_regret]
                        #                         progresses = [progress[:500] for progress in progresses ]
                        sizes = list(map(len, progresses))
                        # more intelligent:
                        max_size = max(sizes)
                        progresses = [
                            np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
                        legend = '{} (mu: {:.3f}, std: {:.5f})'.format(
                            group_legend, best_regret, np.std(best_progress))
                        window_size = np.maximum(
                            int(np.round(max_size / float(1000))), 1)
                        if use_median:
                            percentile25 = np.nanpercentile(
                                progresses, q=25, axis=0)
                            percentile50 = np.nanpercentile(
                                progresses, q=50, axis=0)
                            percentile75 = np.nanpercentile(
                                progresses, q=75, axis=0)
                            if smooth_curve:
                                percentile25 = sliding_mean(percentile25,
                                                            window=window_size)
                                percentile50 = sliding_mean(percentile50,
                                                            window=window_size)
                                percentile75 = sliding_mean(percentile75,
                                                            window=window_size)
                            if clip_plot_value is not None:
                                percentile25 = np.clip(percentile25, -clip_plot_value, clip_plot_value)
                                percentile50 = np.clip(percentile50, -clip_plot_value, clip_plot_value)
                                percentile75 = np.clip(percentile75, -clip_plot_value, clip_plot_value)
                            to_plot.append(
                                ext.AttrDict(percentile25=percentile25, percentile50=percentile50,
                                             percentile75=percentile75, legend=legend_post_processor(legend)))
                        else:
                            means = np.nanmean(progresses, axis=0)
                            stds = np.nanstd(progresses, axis=0)
                            if normalize_error:# and len(progresses) > 0:
                                stds /= np.sqrt(np.sum((1. - np.isnan(progresses)), axis=0))
                            if smooth_curve:
                                means = sliding_mean(means,
                                                     window=window_size)
                                stds = sliding_mean(stds,
                                                    window=window_size)
                            if clip_plot_value is not None:
                                means = np.clip(means, -clip_plot_value, clip_plot_value)
                                stds = np.clip(stds, -clip_plot_value, clip_plot_value)
                            to_plot.append(
                                ext.AttrDict(means=means, stds=stds, legend=legend_post_processor(legend)))
                        if len(to_plot) > 0 and len(data) > 0:
                            to_plot[-1]["footnote"] = "%s; e.g. %s" % (kv_string_best_regret, data[0].params.get("exp_name", "NA"))
                        else:
                            to_plot[-1]["footnote"] = ""
                else:
                    progresses = [
                        exp.progress.get(plot_key, np.array([np.nan])) for exp in filtered_data]
                    sizes = list(map(len, progresses))
                    # more intelligent:
                    max_size = max(sizes)
                    progresses = [
                        np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
                    window_size = np.maximum(int(np.round(max_size / float(1000))), 1)

                    if use_median:
                        percentile25 = np.nanpercentile(
                            progresses, q=25, axis=0)
                        percentile50 = np.nanpercentile(
                            progresses, q=50, axis=0)
                        percentile75 = np.nanpercentile(
                            progresses, q=75, axis=0)
                        if smooth_curve:
                            percentile25 = sliding_mean(percentile25,
                                                        window=window_size)
                            percentile50 = sliding_mean(percentile50,
                                                        window=window_size)
                            percentile75 = sliding_mean(percentile75,
                                                        window=window_size)
                        if clip_plot_value is not None:
                            percentile25 = np.clip(percentile25, -clip_plot_value, clip_plot_value)
                            percentile50 = np.clip(percentile50, -clip_plot_value, clip_plot_value)
                            percentile75 = np.clip(percentile75, -clip_plot_value, clip_plot_value)
                        to_plot.append(
                            ext.AttrDict(percentile25=percentile25, percentile50=percentile50,
                                         percentile75=percentile75, legend=legend_post_processor(legendStr)))
                    else:
                        means = np.nanmean(progresses, axis=0)
                        stds = np.nanstd(progresses, axis=0)
                        if smooth_curve:
                            means = sliding_mean(means,
                                                 window=window_size)
                            stds = sliding_mean(stds,
                                                window=window_size)
                        if clip_plot_value is not None:
                            means = np.clip(means, -clip_plot_value, clip_plot_value)
                            stds = np.clip(stds, -clip_plot_value, clip_plot_value)
                        to_plot.append(
                            ext.AttrDict(means=means, stds=stds, legend=legend_post_processor(legendStr)))

        if len(to_plot) > 0 and not gen_eps:
            fig_title = "%s: %s" % (split_key, split_legend)
            plots.append(make_plot(
                to_plot,
                use_median=use_median, title=fig_title,
                plot_width=plot_width, plot_height=plot_height
            ))

        if gen_eps:
            make_plot_eps(to_plot, use_median=use_median, counter=counter,plot_key=plot_key)
        counter += 1
    return "\n".join(plots)


def parse_float_arg(args, key):
    x = args.get(key, "")
    try:
        return float(x)
    except Exception:
        return None


@app.route("/plot_div")
def plot_div():
    args = flask.request.args
    plot_key = args.get("plot_key")
    if plot_key is None:
        plot_key = "AverageReturn"
    split_key = args.get("split_key", "")
    group_key = args.get("group_key", "")
    if group_key is "":
        group_key = distinct_params[0][0]
    filters_json = args.get("filters", "{}")
    filters = json.loads(filters_json)
    if len(split_key) == 0:
        split_key = None
    if len(group_key) == 0:
        group_key = None

    use_median = args.get("use_median", "") == 'True'
    gen_eps = args.get("eps", "") == 'True'
    only_show_best = args.get("only_show_best", "") == 'True'
    only_show_best_final = args.get("only_show_best_final", "") == 'True'
    only_show_best_sofar = args.get("only_show_best_sofar", "") == 'True'
    normalize_error = args.get("normalize_error", "") == 'True'
    filter_nan = args.get("filter_nan", "") == 'True'
    smooth_curve = args.get("smooth_curve", "") == 'True'
    clip_plot_value = parse_float_arg(args, "clip_plot_value")
    plot_width = parse_float_arg(args, "plot_width")
    plot_height = parse_float_arg(args, "plot_height")
    custom_filter = args.get("custom_filter", None)
    custom_series_splitter = args.get("custom_series_splitter", None)
    if custom_filter is not None and len(custom_filter.strip()) > 0:
        custom_filter = safer_eval(custom_filter)

    else:
        custom_filter = None
    legend_post_processor = args.get("legend_post_processor", None)
    if legend_post_processor is not None and len(legend_post_processor.strip()) > 0:
        legend_post_processor = safer_eval(legend_post_processor)
    else:
        legend_post_processor = None
    if custom_series_splitter is not None and len(custom_series_splitter.strip()) > 0:
        custom_series_splitter = safer_eval(custom_series_splitter)
    else:
        custom_series_splitter = None
    plot_div = get_plot_instruction(plot_key=plot_key, split_key=split_key, filter_nan=filter_nan,
                                    group_key=group_key, filters=filters, use_median=use_median, gen_eps=gen_eps,
                                    only_show_best=only_show_best, only_show_best_final=only_show_best_final,
                                    only_show_best_sofar=only_show_best_sofar,
                                    clip_plot_value=clip_plot_value, plot_width=plot_width, plot_height=plot_height,
                                    smooth_curve=smooth_curve, custom_filter=custom_filter,
                                    legend_post_processor=legend_post_processor, normalize_error=normalize_error,
                                    custom_series_splitter=custom_series_splitter)
    # print plot_div
    return plot_div

def safer_eval(some_string):
    """
    Not full-proof, but taking advice from:

    https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    """
    if "__" in some_string or "import" in some_string:
        raise Exception("string to eval looks suspicious")
    return eval(some_string, {'__builtins__': {}})

@app.route("/time")
def time():
    import matplotlib.pyplot as _plt

    if "Time" in plottable_keys:
        plot_key = "Time"
    elif len(plottable_keys) > 0:
        plot_key = plottable_keys[0]
    else:
        plot_key = None
    if len(distinct_params) > 0:
        group_key = distinct_params[0][0]
    else:
        group_key = None

    group_names = []
    group_data = {}

    column_names = []


    for exp_data in exps_data:
        exp_name = exp_data['params']['log_dir']
        exp_name = exp_name.split("/")
        group_name = exp_name[-2]
        column_name = exp_name[-1].split("_")[0]

        if column_name == 'trpo' or column_name == 'trpo-vime':
            continue

        if group_name not in group_names:
           group_names.append(group_name)
           group_data[group_name] = {}

        if column_name not in column_names:
           column_names.append(column_name)

        if column_name not in group_data[group_name]:
           group_data[group_name][column_name] = []

        t = exp_data['progress'][plot_key]
        t = np.diff(t)
        if len(group_data[group_name][column_name])>0 and len(t) < len(group_data[group_name][column_name][0]):
            continue

        group_data[group_name][column_name].append(t)


    N = len(group_names)
    x = np.arange(N)
    mid = N/2
    width = 1/(N+1)

    fig, ax = _plt.subplots(figsize=(8, 5))

    for j,column_name in enumerate(column_names):
        gt_means = []
        gt_stds = []
        for i,group_name in enumerate(group_names):
           t = np.stack(group_data[group_name][column_name])
           gt_means.append(np.mean(t))
           gt_stds.append(np.std(t))

        rects = ax.bar(x+j*width-(mid-0.5)*width, gt_means,width,yerr=gt_stds,label=column_name,color=color_defaults[j])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_ylabel('Scores')
    #ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.legend()

    if not os.path.exists('figures'):
        os.mkdir('figures')

    _plt.savefig('figures/time.pdf', bbox_inches='tight')

    return "\ndone"


@app.route("/")
def index():
    if "AverageReturn" in plottable_keys:
        plot_key = "AverageReturn"
    elif len(plottable_keys) > 0:
        plot_key = plottable_keys[0]
    else:
        plot_key = None
    if len(distinct_params) > 0:
        group_key = distinct_params[0][0]
    else:
        group_key = None
    plot_div = get_plot_instruction(
        plot_key=plot_key, split_key=None, group_key=group_key)
    return flask.render_template(
        "main.html",
        plot_div=plot_div,
        plot_key=plot_key,
        group_key=group_key,
        plottable_keys=plottable_keys,
        distinct_param_keys=[str(k) for k, v in distinct_params],
        distinct_params=dict([(str(k), list(map(str, v)))
                              for k, v in distinct_params]),
    )

def reload_data():
    global exps_data
    global plottable_keys
    global distinct_params
    exps_data = core.load_exps_data(args.data_paths,args.disable_variant)
    plottable_keys = list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data)))
    plottable_keys = sorted([k for k in plottable_keys if k is not None])
    distinct_params = sorted(core.extract_distinct_params(exps_data))


if __name__ == "__main__":
    #global eps_filename

    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs='*')
    parser.add_argument("--prefix",type=str,nargs='?',default="???")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--disable-variant",default=False,action='store_true')
    args = parser.parse_args(sys.argv[1:])

    # load all folders following a prefix
    if args.prefix != "???":
        args.data_paths = []
        dirname = os.path.dirname(args.prefix)
        subdirprefix = os.path.basename(args.prefix)
        for subdirname in os.listdir(dirname):
            path = os.path.join(dirname,subdirname)
            if os.path.isdir(path) and (subdirprefix in subdirname):
                args.data_paths.append(path)
    print("Importing data from {path}...".format(path=args.data_paths))

    reload_data()
    # port = 5000
    # url = "http://0.0.0.0:{0}".format(port)
    print("Done! View http://localhost:%d in your browser" % args.port)
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
