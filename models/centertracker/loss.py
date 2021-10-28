import tensorflow as tf
from models.centernet.loss import CenternetLoss
from models.centertracker.params import CentertrackerParams


class CentertrackerLoss(CenternetLoss):
    def __init__(self, params: CentertrackerParams):
        super().__init__(params)
        self.params = params

        if params.REGRESSION_FIELDS["track_offset"].active:
            self.track_offset_pos = [params.start_idx("track_offset"), params.end_idx("track_offset")]
        else:
            assert(False and "If you want to use centertracker better make the track_offset available!")

    def track_offset_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.track_offset_pos[0]:self.track_offset_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.track_offset_pos[0]:self.track_offset_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat, loss_type="mse")
        return loss_val

    def call(self, y_true, y_pred):
        total_loss = super().call(y_true, y_pred)
        
        if self.params.REGRESSION_FIELDS["track_offset"].active:
            total_loss += self.track_offset_loss(y_true, y_pred) * self.params.REGRESSION_FIELDS["track_offset"].loss_weight
        
        return total_loss
