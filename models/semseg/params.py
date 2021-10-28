"""
Hold all the specific parameters in this python class
"""

class Params:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 4
        self.PLANED_EPOCHS = 300
        self.LOAD_WEIGHTS = None # Example: "/home/computer-vision-models/trained_models/semseg_comma10k_augment_2021-07-31-125341/tf_model_31/keras.h5"

        # Note that edge tpu compiler currently has very little support for quant aware trainig, and with very little I mean none
        self.QUANTIZE = False
        self.LOAD_WEIGHTS_QUANTIZED = None

        # Index parallel to the label spec (must have same length!), 0: road, 1: lanemarkings, 2: undrivable, 3: movable, 4: ego_car
        self.CLASS_WEIGHTS = [0.9, 1.5, 0.6, 1.3, 0.7]

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img s
        self.MASK_WIDTH = (self.INPUT_WIDTH // 2) # width of the output mask, can only be same size as INPUT_WIDTH or // 2
        self.MASK_HEIGHT = (self.INPUT_HEIGHT // 2) # height of the output mask, can only be same size as INPUT_HEIGHT or // 2
