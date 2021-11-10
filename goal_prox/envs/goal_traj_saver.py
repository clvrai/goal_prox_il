from rlf.il import TrajSaver

class GoalTrajSaver(TrajSaver):
    def __init__(self, save_dir, assert_saved):
        self.assert_saved = assert_saved
        super().__init__(save_dir)

    def should_save_traj(self, traj):
        last_info = traj[-1][-1]
        ret = last_info['ep_found_goal'] == 1.0
        if self.assert_saved and not ret:
            raise ValueError('Trajectory did not end successfully')
        return ret

