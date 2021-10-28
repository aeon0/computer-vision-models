import cv2
from pymongo import MongoClient
import argparse
from tqdm import tqdm
from data.label_spec import Entry
from common.utils import resize_img
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload semseg data from comma10k dataset")
    parser.add_argument("--src_path", type=str, help="Path to comma10k dataset e.g. /home/user/comma10k")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="comma10k", help="MongoDB collection")
    parser.add_argument("--resize", nargs='+', type=int, default=None, help="If set, will resize images and masks to [width, height, offset_bottom]")
    args = parser.parse_args()

    # args.src_path = "/home/jo/training_data/comma10k"
    args.resize = [640, 256, -230]

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    with open(args.src_path + "/files_trainable") as f:
        for trainable_img in tqdm(f.readlines()):
            trainable_img = trainable_img.strip()
            name = trainable_img[6:]
            mask_path = args.src_path + "/" + trainable_img
            img_path = args.src_path + "/imgs/" + name

            if collection.count_documents({'name': name}, limit=1) == 0:
                mask_data = cv2.imread(mask_path)
                img_data = cv2.imread(img_path)
                if args.resize is not None:
                    mask_data, _ = resize_img(mask_data, args.resize[0], args.resize[1], args.resize[2], interpolation=cv2.INTER_NEAREST)
                    img_data, _ = resize_img(img_data, args.resize[0], args.resize[1], args.resize[2])
                    mask_bytes = cv2.imencode('.png', mask_data)[1].tobytes()
                    img_bytes = cv2.imencode('.png', img_data)[1].tobytes()
                    entry = Entry(
                    img=img_bytes,
                    mask=mask_bytes,
                    content_type="image/png",
                    org_source="comma10k",
                    org_id=name,
                    )
                    collection.insert_one(entry.get_dict())

                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                    # ax2.imshow(cv2.cvtColor(mask_data, cv2.COLOR_BGR2RGB))
                    # plt.show()
            else:
                print("WARNING: " + name + " already exist, continue with next image")
