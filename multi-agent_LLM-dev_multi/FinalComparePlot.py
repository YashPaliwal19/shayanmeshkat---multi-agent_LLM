import csv
import matplotlib.pyplot as plt
import numpy as np

def readfile(filepath):
    single_steps = []
    single_rewards = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',') 
        for i in csv_reader:
            single_steps.append(int(i[0]))
            single_rewards.append(int(i[1]))
        # print(f"steps: {single_steps}")
        # print(f"Rewards: {single_rewards}")
    return single_steps, single_rewards

def moving_average(arr, window_size=10):
    moving_arr = []
    for i in range(len(arr)):
        if i<window_size:
            temp = np.sum(arr[:i+1])/(i+1)
        else:
            temp = np.sum(arr[i-window_size+1:i+1])/window_size
        moving_arr.append(temp)

    return moving_arr

def get_percentile_data(all_steps, all_rewards, percentile_list):
    all_rewards = np.array(all_rewards)
    all_steps = np.average(np.array(all_steps), axis=0)

    percentile_dict = {}
    for p in percentile_list:
        percentile_dict[p] = []

    num_rows, num_cols = all_rewards.shape
    # For each column take the percentile
    # Get one column
    for i in range(num_cols):
        rewards_each_episode = all_rewards[:,i]
        for p_val in percentile_list:
            temp = np.percentile(rewards_each_episode, p_val)
            percentile_dict[p_val].append(temp)
    
    # Smooth the percentile data
    for p_val in percentile_list:
        percentile_dict[p_val] = moving_average(percentile_dict[p_val])

    return percentile_dict, all_steps

def modify_data(all_steps, all_rewards, window_size):
    all_steps = np.array(all_steps)
    all_rewards = np.array(all_rewards)
    all_steps = np.average(all_steps, axis=0)
    smooth_trajs = []
    
    for i in range(len(all_rewards)):
        temp = moving_average(all_rewards[i], window_size)
        smooth_trajs.append(temp)

    average_rewards = np.average(smooth_trajs, axis=0)
    #average_rewards = np.array(moving_average(average_rewards, window_size))
    min_rewards = np.amin(smooth_trajs, axis=0)
    #min_rewards = np.array(moving_average(min_rewards, window_size))
    max_rewards = np.max(smooth_trajs, axis=0)
    #max_rewards = np.array(moving_average(max_rewards,window_size))
    print(f"shape of average rewards: {average_rewards.shape}")
    print(f"shape of minimum rewards: {min_rewards.shape}")
    print(f"shape of maximum rewards: {max_rewards.shape}")
    return average_rewards, min_rewards, max_rewards, all_steps

def SmallOfficeWorldParams():
    step_unit = 85
    percentile_list = [10,25,50,75,90]
    num_times = 10
    all_steps_wotlcd = []
    all_rewards_wotlcd = []
    all_steps_wtlcd = []
    all_rewards_wtlcd = []
    
    for i in range(num_times):
        filepath = f"yashOfficeWorld/SmallTaskDFA_FinalPaper_WOTLCD_{i}.csv"
        single_steps, single_rewards = readfile(filepath)
        all_steps_wotlcd.append(single_steps)
        all_rewards_wotlcd.append(single_rewards)

    # Getting data with TLCD
    for i in range(num_times):
        filepath = f"yashOfficeWorld/SmallTaskDFA_FinalPaper_WTLCD_{i}.csv"
        single_steps, single_rewards = readfile(filepath)
        all_steps_wtlcd.append(single_steps)
        all_rewards_wtlcd.append(single_rewards)
    
    cut_num = 185
    for i in range(num_times):
        print(f"In loop {i}")
        all_steps_wotlcd[i] = all_steps_wotlcd[i][:cut_num]
        all_steps_wtlcd[i] = all_steps_wtlcd[i][:cut_num]
        all_rewards_wotlcd[i] = all_rewards_wotlcd[i][:cut_num]
        all_rewards_wtlcd[i] = all_rewards_wtlcd[i][:cut_num]

    print(f"length: {len(all_steps_wotlcd[0][:cut_num])}")
    
    return all_steps_wotlcd, all_rewards_wotlcd, all_steps_wtlcd, all_rewards_wtlcd, percentile_list, step_unit


def LargeOfficeWorldParams():
    step_unit = 250
    percentile_list = [10,25,50,75,90]
    num_times = 10
    all_steps_wotlcd = []
    all_rewards_wotlcd = []
    all_steps_wtlcd = []
    all_rewards_wtlcd = []
    
    for i in range(num_times):
        filepath = f"yashOfficeWorld/LargeTaskDFA_FinalPaper_WOTLCD_{i}.csv"
        single_steps, single_rewards = readfile(filepath)
        all_steps_wotlcd.append(single_steps)
        all_rewards_wotlcd.append(single_rewards)

    # Getting data with TLCD
    for i in range(num_times):
        filepath = f"yashOfficeWorld/LargeTaskDFA_FinalPaper_WTLCD_{i}.csv"
        single_steps, single_rewards = readfile(filepath)
        all_steps_wtlcd.append(single_steps)
        all_rewards_wtlcd.append(single_rewards)
    
    cut_num = 530
    for i in range(num_times):
        print(f"In loop {i}")
        all_steps_wotlcd[i] = all_steps_wotlcd[i][:cut_num]
        all_steps_wtlcd[i] = all_steps_wtlcd[i][:cut_num]
        all_rewards_wotlcd[i] = all_rewards_wotlcd[i][:cut_num]
        all_rewards_wtlcd[i] = all_rewards_wtlcd[i][:cut_num]

    # print(f"length: {len(all_steps_wotlcd[0][:cut_num])}")

    return all_steps_wotlcd, all_rewards_wotlcd, all_steps_wtlcd, all_rewards_wtlcd, percentile_list, step_unit


def CrossRoadWorldParams():
    step_unit = 500
    percentile_list = [10,25,50,75,90]
    num_times = 10
    all_steps_wotlcd = []
    all_rewards_wotlcd = []
    all_steps_wtlcd = []
    all_rewards_wtlcd = []
    
    for i in range(num_times):
        filepath = f"yashOfficeWorld/ButtonWorld_FinalPaper_WOTLCD_{i}.csv"
        single_steps, single_rewards = readfile(filepath)
        all_steps_wotlcd.append(single_steps)
        all_rewards_wotlcd.append(single_rewards)

    # Getting data with TLCD
    for i in range(num_times):
        filepath = f"yashOfficeWorld/ButtonWorld_FinalPaper_WTLCD_{i}.csv"
        single_steps, single_rewards = readfile(filepath)
        all_steps_wtlcd.append(single_steps)
        all_rewards_wtlcd.append(single_rewards)

    cut_num = 40
    for i in range(num_times):
        print(f"In loop {i}")
        all_steps_wotlcd[i] = all_steps_wotlcd[i][:cut_num]
        all_steps_wtlcd[i] = all_steps_wtlcd[i][:cut_num]
        all_rewards_wotlcd[i] = all_rewards_wotlcd[i][:cut_num]
        all_rewards_wtlcd[i] = all_rewards_wtlcd[i][:cut_num]

    # print(f"length: {len(all_steps_wotlcd[0][:cut_num])}")
    
    return all_steps_wotlcd, all_rewards_wotlcd, all_steps_wtlcd, all_rewards_wtlcd, percentile_list, step_unit



if __name__=="__main__":
    num_times = 10
    window_size = 10
    all_steps_wotlcd, all_rewards_wotlcd, all_steps_wtlcd, all_rewards_wtlcd, percentile_list, step_unit = CrossRoadWorldParams()
    # step_unit =CrossRoadWorldParams()    # percentile_list = [5,25,50,75,95]

    # Getting data without TLCD
    # for i in range(num_times):
    #     filepath = f"yashOfficeWorld/ButtonWorld_FinalPaper_WTLCD_{i}.csv"
    #     single_steps, single_rewards = readfile(filepath)
    #     all_steps_wotlcd.append(single_steps)
    #     all_rewards_wotlcd.append(single_rewards)

    # # Getting data with TLCD
    # for i in range(num_times):
    #     filepath = f"yashOfficeWorld/ButtonWorld_FinalPaper_WOTLCD_{i}.csv"
    #     single_steps, single_rewards = readfile(filepath)
    #     all_steps_wtlcd.append(single_steps)
    #     all_rewards_wtlcd.append(single_rewards)


    # get the percentile data and smooth data inside
    percen_data_wtlcd, steps_per_wtlcd = get_percentile_data(all_steps_wtlcd, all_rewards_wtlcd, percentile_list)
    percen_data_wotlcd, steps_per_wotlcd = get_percentile_data(all_steps_wotlcd, all_rewards_wotlcd, percentile_list)
    
    # Get averaged out data
    average_rewards_wotlcd, min_rewards_wotlcd, max_rewards_wotlcd, all_steps_wotlcd = modify_data(all_steps_wotlcd, all_rewards_wotlcd, window_size)
    average_rewards_wtlcd, min_rewards_wtlcd, max_rewards_wtlcd, all_steps_wtlcd = modify_data(all_steps_wtlcd, all_rewards_wtlcd, window_size)

    total_steps = [x for x in all_steps_wotlcd.shape]
    print(f"Total steps: {total_steps}")
    # Setting up Figure        
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    plt.xlim(0, (total_steps[0]-1)*step_unit)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)


    plt.plot(steps_per_wtlcd, percen_data_wtlcd[50], 'r', alpha=1, label="With TL-CD")
    plt.fill_between(steps_per_wtlcd, percen_data_wtlcd[10], percen_data_wtlcd[90], facecolor='red', alpha=0.05)
    plt.fill_between(steps_per_wtlcd, percen_data_wtlcd[25], percen_data_wtlcd[75], facecolor='red', alpha=0.25)
    plt.plot(steps_per_wotlcd, percen_data_wotlcd[50], 'g', alpha=1, label="Without TL-CD")
    plt.fill_between(steps_per_wotlcd, percen_data_wotlcd[10], percen_data_wotlcd[90], facecolor='green', alpha=0.05)
    plt.fill_between(steps_per_wotlcd, percen_data_wotlcd[25], percen_data_wotlcd[75], facecolor='green', alpha=0.25)

    # plt.plot(all_steps_wtlcd, average_rewards_wtlcd,'r',alpha=1, label="With TL-CD")
    # plt.fill_between(all_steps_wtlcd, min_rewards_wtlcd, max_rewards_wtlcd, color='red', alpha=0.25)
    # plt.plot(all_steps_wotlcd, average_rewards_wotlcd,'g', alpha=1, label="Without TL-CD")
    # plt.fill_between(all_steps_wotlcd, min_rewards_wotlcd, max_rewards_wotlcd, color='green', alpha=0.25)

    plt.xlabel("Number of steps")
    plt.ylabel("Reward")
    plt.title("Performance Comparison")
    plt.legend(loc='lower right')
    
    
    plt.show()

