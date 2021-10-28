from models.centernet.params import CenternetParams

class CentertrackerParams(CenternetParams):
    def __init__(self, nb_classes):
        super().__init__(nb_classes)

        # Training
        self.BATCH_SIZE = 8
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH_BASE = None # Path to a centernet for the base model
        self.LOAD_PATH = None # Path to a complete center tracker

        # Params for t-1 frame objects
        self.FN_PROB = 0.00 # Probability of FN per object
        self.FP_PROB = 0.00 # Probability of FP per frame
        self.POS_NOISE_WEIGHT = 0.00 # Weight for gaus noise that changes ground truth object position

        # Add the track offset to the regression fields, should always be set to True otherwise CenterTrack does not make much sense
        self.REGRESSION_FIELDS["track_offset"] = CenternetParams.RegressionField(True, 2, 0.1, "x and y offset to track at t-1 relative to input size")

    def serialize(self):
        dict_data = super().serialize()
        dict_data["LOAD_PATH_BASE"] = self.LOAD_PATH_BASE
        return dict_data
