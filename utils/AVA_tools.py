import csv
import os
import argparse
import pandas as pd
from PIL import Image


def test_tool():
    ava_csv = '/home/lab325/gry/gry-graduation/style-binary-classification/data/2_style_binary_train.csv'
    ava_df = pd.DataFrame(pd.read_csv(ava_csv, header=None))
    ava_df[0] = ava_df[0].astype(int)

    ava_df.to_csv(ava_csv, index=False, header=False)


def get_csv_file_name(low, high, mode=None):
    if mode:
        return mode + '_' + str(low).replace('.', '') + '_' + str(high).replace('.', '') + '.csv'
    else:
        name = str(low).replace('.', '') + '_' + str(high).replace('.', '') + '.csv'
        return name


def save_train_val_test_csv(train_prop, val_prop, test_prop, csv_path, csv_name, save_split_csv_path):
    df = pd.read_csv(csv_path + csv_name, header=None)
    df[0] = df[0].astype(int)
    if csv_name == 'no_style.csv':
        df[11] = 0
        df[12] = 1
    else:
        df[11] = 1
        df[12] = 0
    base_file_name = csv_name.split('.')[0]

    train_num = int(df.shape[0] * train_prop)
    val_end = train_num + int(df.shape[0] * val_prop)
    test_num = int(df.shape[0] * test_prop)

    if train_prop > 0:
        if not os.path.exists(save_split_csv_path + 'train'):
            os.makedirs(save_split_csv_path + 'train')
        train = pd.DataFrame(df.head(train_num), index=None, columns=None)
        train.to_csv(save_split_csv_path + 'train/' + base_file_name + "_train.csv", index=False, header=False)
    if val_prop > 0:
        if not os.path.exists(save_split_csv_path + 'val'):
            os.makedirs(save_split_csv_path + 'val')
        val = pd.DataFrame(df.iloc[train_num:val_end, :])
        val.to_csv(save_split_csv_path + 'val/' + base_file_name + "_val.csv", index=False, header=False)
    if test_prop > 0:
        if not os.path.exists(save_split_csv_path + 'test'):
            os.makedirs(save_split_csv_path + 'test')
        test = pd.DataFrame(df.tail(test_num + 1), index=None, columns=None)
        test.to_csv(save_split_csv_path + 'test/' + base_file_name + "_test.csv", index=False, header=False)


def clean_dataset_csv(csv_path, csv_name, img_path):
    with open(csv_path + str(csv_name).replace('.', '_') + '_error_info.txt', 'wt') as err_txt:
        with open(csv_path + csv_name, 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                img = img_path + str(row[0]) + '.jpg'
                if os.path.exists(img):
                    try:
                        img = Image.open(img)
                    except ValueError as err:
                        print(err)
                        err_txt.write('Corrupt Image:' + str(row) + '\n')
                else:
                    err_txt.write('Not Found Image:' + str(row) + '\n')
    print('clean')


def get_section_csv(low, high, write_in_path, img_path, csv_file):
    with open(write_in_path + get_csv_file_name(low, high), 'wt') as f1:
        cw = csv.writer(f1)
        with open(csv_file, 'rt') as f2:
            reader = csv.reader(f2, delimiter=',')
            for row in reader:
                img = img_path + str(row[0]) + '.jpg'
                img = Image.open(img)
                img_wh_ratio = img.size[0] / img.size[1]
                if low < img_wh_ratio < high:
                    cw.writerow(row)


def get_train_val_test(train_prop, val_prop, test_prop, csv_path, csv_name, batch_split, save_split_csv_path):
    if batch_split:     # 批量分割训练验证测试集
        for file in os.listdir(csv_path):
            if file.split('.')[-1] == 'csv':
                save_train_val_test_csv(train_prop, val_prop, test_prop, csv_path, file, save_split_csv_path)
    else:
        save_train_val_test_csv(train_prop, val_prop, test_prop, csv_path, csv_name, save_split_csv_path)

def get_style_classification_train_val_test(csv_path):
    for root, dirs, files in os.walk(csv_path):
        for dir_name in dirs:
            no_style_file_name = os.path.join(csv_path, dir_name, 'no_style_' + dir_name + '.csv')
            no_style_df = pd.DataFrame(pd.read_csv(no_style_file_name, index_col=0, header=None))
            print(no_style_file_name)
            for file in os.listdir(os.path.join(csv_path, dir_name)):
                if file.split('.')[-1] != 'csv' or file.split('.')[0] == 'no_style_' + dir_name:
                    continue
                print(file)
                style_df = pd.DataFrame(pd.read_csv(os.path.join(csv_path, dir_name, file), index_col=0, header=None))
                print(style_df.shape)
                append_df = no_style_df.sample(len(style_df))
                print(append_df.shape)
                style_df = style_df.append(append_df, ignore_index=False)
                print(style_df.shape)
                style_df = style_df.sample(frac=1)
                style_df.to_csv(os.path.join(csv_path, dir_name, file), header=None)


def get_no_style_images():
    ava_csv = '/home/lab325/gry/Neural-IMage-Assessment/data/AVA_score_percent.csv'
    ava_df = pd.DataFrame(pd.read_csv(ava_csv, index_col=0, header=None))
    style_csv_path = '/home/lab325/gry/Neural-IMage-Assessment/data/style_csvs/'

    for i in range(1, 15):
        style_df = pd.DataFrame(pd.read_csv(style_csv_path + str(i) + '.csv', index_col=None, header=None))
        print('now ' + str(i) + '.csv')

        for j in range(len(style_df)):
            try:
                ava_df = ava_df.drop(index=[style_df[0][j]])
            except:
                print(style_df[0][j])

    ava_df.to_csv(style_csv_path + 'no_style.csv', header=None)



# 合并正负样本
def merge_samples(positive_csv, negative_csv, save_path, merge_type):
    positive_df = pd.read_csv(positive_csv, header=None)
    negative_df = pd.read_csv(negative_csv, header=None)
    merge_df = pd.DataFrame()

    if merge_type == 'train':
        print('merge train csv file')
        copy_time = int(len(negative_df) / len(positive_df))
        remainder = len(negative_df) % len(positive_df)

        for i in range(copy_time):
            merge_df = merge_df.append(positive_df)

        for i in range(remainder):
            a = positive_df.loc[i]
            d = pd.DataFrame(a).T
            merge_df = merge_df.append(d)

        merge_df = merge_df.append(negative_df)
        merge_df = merge_df.sample(frac=1)
        merge_df[0] = merge_df[0].astype(int)

        print('positive_df.shape', positive_df.shape)
        print('negative_df.shape', negative_df.shape)
        print('copy_time', copy_time)
        print('remainder', remainder)
        print('merge_df.shape', merge_df.shape)
        print('save as', save_path)
        merge_df.to_csv(save_path, index=False, header=False)

    if merge_type == 'val':
        print('merge validation csv file')
        negative_df = negative_df.sample(len(positive_df))
        merge_df = merge_df.append(positive_df)
        merge_df = merge_df.append(negative_df)
        merge_df = merge_df.sample(frac=1)
        print('positive_df.shape', positive_df.shape)
        print('negative_df.shape', negative_df.shape)
        print('merge_df.shape', merge_df.shape)
        print('save as', save_path)
        merge_df.to_csv(save_path, index=False, header=False)





def main(args):
    if args.test_tool:
        test_tool()


    if args.clean_dataset_csv:
        clean_dataset_csv(args.clean_csv_path,
                          args.clean_csv_name,
                          args.clean_img_path)

    if args.get_section:
        for i in range(args.iteration_num):
            get_section_csv(round(args.start_ratio + i*0.1, 2),
                            round(args.end_ratio + i*0.1, 2),
                            args.save_section_csv_path, args.img_path,
                            args.init_csv_path)

    if args.get_train_val_test:
        get_train_val_test(args.train_proportion,
                           args.val_proportion,
                           args.test_proportion,
                           args.split_csv_path,
                           args.split_csv_name,
                           args.batch_split,
                           args.save_split_csv_path)
    if args.get_no_style_images:
        get_no_style_images()

    if args.get_style_classification_train_val_test:
        get_style_classification_train_val_test(args.style_split_csv_path)


    if args.merge_samples:
        merge_samples(args.positive_csv,
                      args.negative_csv,
                      args.save_merge_csv_path,
                      args.merge_type)


if __name__ == '__main__':
    # todo 可以加一个判断path最后是否是‘/’，如果不是则拼接‘/’
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_tool', action='store_true')


    # clean_dataset_csv
    parser.add_argument('--clean_dataset_csv', action='store_true')
    parser.add_argument('--clean_csv_path', type=str, default='../CSV/')
    parser.add_argument('--clean_csv_name', type=str, default='AVA_score_percent.csv')
    parser.add_argument('--clean_img_path', type=str, default='/home/lab325/Download/AVA_dataset/images/images/')

    # get_section_csv
    parser.add_argument('--get_section', action='store_true')
    parser.add_argument('--start_ratio', type=float, default=0.3)
    parser.add_argument('--end_ratio', type=float, default=0.4)
    parser.add_argument('--iteration_num', type=int, default=20)
    parser.add_argument('--save_section_csv_path', type=str, default='../CSV/train_val_test_csv/train_section_csv_2/')
    parser.add_argument('--img_path', type=str, default='/home/lab325/Download/AVA_dataset/images/images/')
    parser.add_argument('--init_csv_path', type=str, default='../CSV/train_val_test_csv/AVA_score_percent_train.csv')

    # get_train_val_test_csv
    parser.add_argument('--get_train_val_test', action='store_true')
    parser.add_argument('--batch_split', type=bool, default=True)
    parser.add_argument('--save_split_csv_path', type=str, default='/home/lab325/gry/gry-graduation/style-binary-classification/data/')
    parser.add_argument('--split_csv_path', type=str, default='/home/lab325/gry/Innovative/AVA_dataset/style-class/style_csvs/')
    parser.add_argument('--split_csv_name', type=str, default='no_style.csv')
    parser.add_argument('--train_proportion', type=float, default=0.9)
    parser.add_argument('--val_proportion', type=float, default=0.1)
    parser.add_argument('--test_proportion', type=float, default=0)


    # AVA dataset process utils
    parser.add_argument('--get_no_style_images', action='store_true')

    # get_style_classification_train_val_test
    parser.add_argument('--get_style_classification_train_val_test', action='store_true')
    parser.add_argument('--style_split_csv_path', type=str, default='/home/lab325/gry/Neural-IMage-Assessment/data/style_csvs/')

    # merge positive and negative samples
    parser.add_argument('--merge_samples', action='store_true')
    parser.add_argument('--positive_csv', type=str, default='/home/lab325/gry/gry-graduation/style-binary-classification/data/train/2_train.csv')
    parser.add_argument('--negative_csv', type=str, default='/home/lab325/gry/gry-graduation/style-binary-classification/data/train/no_style_train.csv')
    parser.add_argument('--save_merge_csv_path', type=str, default='/home/lab325/gry/gry-graduation/style-binary-classification/data/2_style_binary_train.csv')
    parser.add_argument('--merge_type', type=str, default='train')


    arguments = parser.parse_args()
    main(arguments)
