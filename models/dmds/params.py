class DmdsParams:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 2
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH = None
        self.LOAD_DEPTH_MODEL = None

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img size

        # Loss weights
        self.rgb_cons = (1.5 / 255.0)
        self.ssim_cons = 2.5
        self.depth_cons = 0.0
        self.supervise_depth = 10.0
        self.depth_smoothing = 0.002
        self.var_depth = 1e-6
        self.rot_cyc = 5.0e-3
        self.tran_cyc = 5.0e-2
        self.mot_smoothing = 1e-4
        self.mot_drift = 1e-6
