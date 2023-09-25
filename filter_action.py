def filter_and_save_lines(input_file, output_file, keyword_lst):
    try:
        with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:
            for line in f_input:
                for keyword in keyword_lst:
                    if keyword in line:
                        f_output.write(line)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    name_str = "1017_seed_50"
    input_file_path = "/sciclone/pscr/pwang12/log_5/8_25/logdir/"+name_str+".log"  # 输入的 log 文件路径
    output_file_path = "/sciclone/pscr/pwang12/log_5/8_25/logdir/"+name_str+"_filtered.log"  # 输出的保存包含特定字符行的文件路径
    keyword_to_search = ["curr seed set",
                         "simulation: seed set",
                         "train - get reward is",
                         "agent.py - update - batch length is",
                         "iteration,  training in graph",
                         "after main agent update, get loss :",
                         "after budget selection, main agent's return is"]  # 要搜索的特定字符或关键词

    filter_and_save_lines(input_file_path, output_file_path, keyword_to_search)
    print("Filtering and saving complete.")
