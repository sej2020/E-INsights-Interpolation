import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

### Global Style Parameters ###
##############################
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
CHONK_SIZE = 24
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:red")
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:red") 
sns.set_style("darkgrid", {'font.family':['serif'], 'axes.edgecolor':'red','ytick.left': True})
plt.ticklabel_format(style = 'plain')
plt.rcParams['axes.edgecolor'] = 'black'
###############################


def histogram_results(
        yaml_list: list, 
        model_name_list: list, 
        model_colors_list: list, 
        int_len_list: list, 
        repetitions: int, 
        out_folder: str
        ):
    """
    Creates histograms of the distribution of model RMSEs for an instance of the direct evaluations experiment. 

    Args:
        yaml_list: list of strings, each string is the path to a yaml file containing the results of a model's 
            direct evaluations
        model_name_list: list of strings, each string is the name of a model corresponding to the yaml file at the 
            same index in yaml_list
        model_colors_list: list of strings, each string is a color to be used in the histogram for the model at the 
            same index in model_name_list
        int_len_list: list of integers, each integer is an interval length for which to create a histogram
        repetitions: the number of repetitions of each interval length in the direct evaluation
        out_folder: the name of the folder in which to save the histograms
    """
    model_dict = {}
    for yam, mod_name in zip(yaml_list, model_name_list):
        with open(yam, 'r') as stream_1:
            try:
                results = yaml.safe_load(stream_1)
            except yaml.YAMLError as exc:
                print(exc)

        result_dict = {}
        for _, file_res in results.items():
            _, _, rmses = file_res.values()
            for len_idx, int_len in enumerate(int_len_list):
                result_dict[int_len] = rmses[len_idx*repetitions : len_idx*repetitions + repetitions] # might throw error at end of list ?

        model_dict[mod_name] = result_dict

    plt.clf()   
    for int_len in int_len_list:
        for color, mod_name in zip(model_colors_list, model_name_list):
            sns.kdeplot(model_dict[mod_name][int_len], label=mod_name, color=color, bw_adjust=0.5)
        plt.legend()
        plt.xlabel("RMSE (Normalized)")
        plt.ylabel("Frequency")
        plt.title(f"RMSE Distribution on Interval Length of {int_len}m")
        pathlib.Path(f"plots/{out_folder}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"plots/{out_folder}/dens_{int_len}s.png")
        plt.clf()



def bar_chart_comp(
        yaml_list: list, 
        model_name_list: list, 
        model_colors_list: list, 
        int_len_list: list, 
        repetitions: int, 
        out_folder: str
        ):
    """
    Creates a grouped bar chart comparing the average RMSEs of models for different interval lengths in an instance of 
    the direct evaluations experiment.

    Args:
        yaml_list: list of strings, each string is the path to a yaml file containing the results of a model's 
            direct evaluations
        model_name_list: list of strings, each string is the name of a model corresponding to the yaml file at the 
            same index in yaml_list
        model_colors_list: list of strings, each string is a color to be used in the bar chart for the model at the 
            same index in model_name_list
        int_len_list: list of integers, each integer is an interval length for which a bar group will be created in 
            the chart
        repetitions: the number of repetitions of each interval length in the direct evaluation
        out_folder: the name of the folder in which to save the grouped bar chart
    """
    model_dict = {}
    for yam, mod_name in zip(yaml_list, model_name_list):
        with open(yam, 'r') as stream_1:
            try:
                results = yaml.safe_load(stream_1)
            except yaml.YAMLError as exc:
                print(exc)

        result_dict = {}
        for _, file_res in results.items():
            _, _, rmses = file_res.values()
            for len_idx, int_len in enumerate(int_len_list):
                result_dict[int_len] = np.nansum(np.array(rmses[len_idx*repetitions : len_idx*repetitions + repetitions])).item()/repetitions # might throw error at end of list ?

        model_dict[mod_name] = result_dict
    plt.clf()
    bar_dict = {}
    for mod_name in model_name_list:
        bar_dict[mod_name] = [model_dict[mod_name][int_len] for int_len in int_len_list]
    print(bar_dict)
    #grouped bar chart
    barWidth = 0.2
    r1 = np.arange(len(int_len_list))
    for bars, color, mod_name in zip(bar_dict.values(), model_colors_list, model_name_list):
        plt.bar(r1, bars, color=color, width=barWidth, edgecolor='white', label=mod_name)
        r1 = [x + barWidth for x in r1]
    plt.xlabel('Interval Length')
    plt.xticks([r + barWidth/2*(len(model_name_list)-1) for r in range(len(int_len_list))], [str(int_len) for int_len in int_len_list])
    plt.ylabel('Average RMSE (Normalized)')
    plt.title('Average RMSE by Interval Length (min)')
    plt.legend()
    pathlib.Path(f"plots/{out_folder}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{out_folder}/bar_comp.png")
    plt.clf()
   
    
if __name__ == "__main__":    

    for data in ["CNC", "HVAC", "M00", "M02"]:
        yamls = [f"output/eval_{data}_baseline.yaml", f"output/eval_{data}_backwardsLSTM.yaml", f"output\eval_{data}_LSTM.yaml"]
        names = ["Baseline", "LSTM (rev)", "LSTM"]
        colors = ["red", "seagreen", "blue"]
        len_list = [15, 60, 360]
        rep = 500
        out_folder = f"eval/{data}"

        histogram_results(
            yaml_list=yamls, 
            model_name_list=names, 
            model_colors_list=colors, 
            int_len_list=len_list, 
            repetitions=rep, 
            out_folder=out_folder
            )
        
        bar_chart_comp(
            yaml_list=yamls, 
            model_name_list=names, 
            model_colors_list=colors, 
            int_len_list=len_list, 
            repetitions=rep, 
            out_folder=out_folder
            )