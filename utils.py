import mne
import pymatreader
import numpy as np



ch_names = [
    "Fp1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "Iz",
    "Oz",
    "POz",
    "Pz",
    "CPz",
    "Fpz",
    "Fp2",
    "AF8",
    "AF4",
    "AFz",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCz",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
]


class Subject:
    def __init__(self, subject_number, montage="biosemi64"):
        folder = "EEG_Data/"
        trial_path = folder + "s" + str(subject_number).zfill(2) + ".mat"
        ch_types = ["eeg" for i in range(64)]
        self.rawdata = pymatreader.read_mat(trial_path)
        self.name = self.rawdata["eeg"]["subject"]
        self.n_trials = self.rawdata["eeg"]["n_imagery_trials"]
        self.noise = self.rawdata["eeg"]["noise"]
        self.rest = self.rawdata["eeg"]["rest"]
        self.srate = self.rawdata["eeg"]["srate"]

        self.imagery_left = self.rawdata["eeg"]["imagery_left"]
        self.imagery_right = self.rawdata["eeg"]["imagery_right"]

        self.imagery_event = self.rawdata["eeg"]["imagery_event"]

        # self.locations = self.rawdata["eeg"][
        #     "senloc"
        # ]  # Locations are unfortunately useless without the reference points
        self.locations_unit_sphere = self.rawdata["eeg"]["psenloc"]

        self.montage = mne.channels.make_standard_montage(montage)
        self.info = mne.create_info(
            ch_names, self.rawdata["eeg"]["srate"], ch_types=ch_types
        ).set_montage(self.montage)
        self.raw_array_left = mne.io.RawArray(self.imagery_left[:64], self.info)
        # self.raw_array_left.set_montage(self.montage)
        self.raw_array_right = mne.io.RawArray(self.imagery_right[:64], self.info)
        # self.raw_array_right.set_montage(self.montage)
        self.steps = list()

    def get_rawdata(self) -> list[mne.io.RawArray]:
        return [self.raw_array_left.copy(), self.raw_array_right.copy()]

    def get_processed_data(self) -> list[mne.io.RawArray]:
        if isinstance(self.processed_raw_left, type(None)):
            print("Preprocessing data...")
            self.preprocess()
        return [self.processed_raw_left.copy(), self.processed_raw_right.copy()]

    def get_epochs(self) -> list[mne.Epochs]:
        if isinstance(self.epochs_left, type(None)):
            print("Preprocessing data...")
            self.preprocess()
        return [self.epochs_left.copy(), self.epochs_right.copy()]

    def preprocess(self, steps={"average_reference": []}):
        self.steps = list()
        for step in steps.keys():
            match step:
                case "average_reference":
                    self.processed_raw_left = (
                        self.raw_array_left.copy().set_eeg_reference(
                            ref_channels="average", projection=True
                        )
                    )
                    self.processed_raw_right = (
                        self.raw_array_right.copy().set_eeg_reference(
                            ref_channels="average", projection=True
                        )
                    )
                    self.steps.append("average_eeg_reference")
                    self.processed_raw_left.apply_proj()
                    self.processed_raw_right.apply_proj()
                    print("Applied average reference")
                case "rest_projection":
                    rest_raw_array = mne.io.RawArray(self.rest[:64], self.info)
                    rest_proj = mne.compute_proj_raw(
                        rest_raw_array, n_eeg=3, n_mag=0, n_grad=0, verbose="DEBUG"
                    )
                    self.processed_raw_right.add_proj(rest_proj)
                    self.processed_raw_left.add_proj(rest_proj)
                    self.steps.append("add_rest_proj")
                case "artifact_projection":
                    noise_dict = {
                        "eye_blinking": 0,
                        "eyeball_movement_up_down": 1,
                        "eyeball_movement_left_right": 2,
                        "jaw_clenching": 3,
                        "head_movement_left_right": 4,
                    }
                    for artifact in steps["artifact_projection"]:
                        noise_raw_array = mne.io.RawArray(
                            self.noise[noise_dict[artifact]][:64], self.info
                        )
                        noise_proj = mne.compute_proj_raw(
                            noise_raw_array,
                            n_eeg=3,
                            n_mag=0,
                            n_grad=0,
                            verbose="DEBUG",
                        )
                        steps.append("add_" + artifact + "_proj")
                        self.processed_raw_right.add_proj(noise_proj)
                        self.processed_raw_left.add_proj(noise_proj)
                case "ICA":
                    ica = mne.preprocessing.ICA(
                        n_components=steps["ICA"]["n_components"],
                        method="fastica",
                        random_state=0,
                        max_iter=1000,
                        verbose="DEBUG",
                    )
                    ica.fit(self.processed_raw_left)
                    self.processed_raw_left = ica.apply(self.processed_raw_left)
                    ica.fit(self.processed_raw_right)
                    self.processed_raw_right = ica.apply(self.processed_raw_right)
                    self.steps.append("ICA")
                case "bandpass_filter":
                    self.processed_raw_left.filter(
                        steps["bandpass_filter"][0], steps["bandpass_filter"][1]
                    )
                    self.steps.append(
                        f"bandpass_filter_{steps['bandpass_filter'][0]}_{steps['bandpass_filter'][1]}"
                    )

        event_indexes = []
        event_id = {"imagery_event": 1}
        for i in range(len(self.imagery_event)):
            if self.imagery_event[i] == 1:
                event_indexes.append(i)
        events = np.array(
            [event_indexes, [0] * len(event_indexes), [1] * len(event_indexes)]
        ).T
        self.epochs_left = mne.Epochs(
            self.processed_raw_left,
            events,
            tmin=-2,
            tmax=5,
            baseline=(-0.5, -0.2),
            event_id=event_id,
        )
        self.epochs_right = mne.Epochs(
            self.processed_raw_right,
            events,
            tmin=-2,
            tmax=5,
            baseline=(-0.5, -0.2),
            event_id=event_id,
        )
        self.steps.append("create_epochs")
        # TODO: Is this indexation really correct?
        # Don't know if [...][0] is left or right
        bad_epochs_left = [
            x - 1
            for x in list(
                self.rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][0]
            )
            + list(self.rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][0])
        ]
        bad_epochs_right = [
            x - 1
            for x in list(
                self.rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][1]
            )
            + list(self.rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][1])
        ]

        self.epochs_left.drop(bad_epochs_left)
        self.epochs_right.drop(bad_epochs_right)
        self.steps.append("drop_bad_trials")


def remove_artifacts(epochs, ica, threshold, t_end, picks, reason="artifacts"):
    source = ica.get_sources(epochs.load_data())
    epoch_indices = []
    for i in range(len(source)):
        cropped_data = source[i].pick(picks).copy().crop(None, t_end)
        # if max value in cropped data exceeds threshold, drop epoch
        if cropped_data._data.max() > threshold:
            # selected_epochs.append(source[i])
            epoch_indices.append(i)
    return epochs.copy().drop(epoch_indices, reason=reason)


def make__raw_and_epochs(
    subject_number, set_average_reference=True, filter=None, baseline=None, reject=None
):
    ch_types = ["eeg" for i in range(64)]
    rawdata = pymatreader.read_mat("EEG_Data/s" + str(subject_number).zfill(2) + ".mat")

    imagery_left = rawdata["eeg"]["imagery_left"]
    imagery_right = rawdata["eeg"]["imagery_right"]
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(
        ch_names, rawdata["eeg"]["srate"], ch_types=ch_types
    ).set_montage(montage)

    raw_array_left = mne.io.RawArray(imagery_left[:64] * 1e-9, info)
    raw_array_right = mne.io.RawArray(imagery_right[:64] * 1e-9, info)

    if set_average_reference:
        raw_array_left.set_eeg_reference(
            ref_channels="average", projection=True
        ).apply_proj()
        raw_array_right.set_eeg_reference(
            ref_channels="average", projection=True
        ).apply_proj()
    if filter is not None:
        raw_array_left.filter(filter[0],filter[1])
        raw_array_right.filter(filter[0],filter[1])

    imagery_event = rawdata["eeg"]["imagery_event"]
    event_id_left = {"imagery_left": 1}
    event_id_right = {"imagery_right": 2}
    event_indexes = []
    for i in range(len(imagery_event)):
        if imagery_event[i] == 1:
            event_indexes.append(i)
    events_left = np.array(
        [event_indexes, [0] * len(event_indexes), [1] * len(event_indexes)]
    ).T
    events_right = np.array(
        [event_indexes, [0] * len(event_indexes), [2] * len(event_indexes)]
    ).T

    if baseline:
        epochs_left = mne.Epochs(
            raw_array_left,
            events_left,
            tmin=-2,
            tmax=5,
            baseline=baseline,
            event_id=event_id_left,
            reject=reject,
        )
        epochs_right = mne.Epochs(
            raw_array_right,
            events_right,
            tmin=-2,
            tmax=5,
            baseline=baseline,
            event_id=event_id_right,
            reject=reject,
        )
    else:
        epochs_left = mne.Epochs(
            raw_array_left,
            events_left,
            tmin=-2,
            tmax=5,
            event_id=event_id_left,
            baseline=None,
            reject=reject,
        )
        epochs_right = mne.Epochs(
            raw_array_right,
            events_right,
            tmin=-2,
            tmax=5,
            event_id=event_id_right,
            baseline=None,
            reject=reject,
        )
    # TODO: Is this indexation really correct?
    # Don't know if [...][0] is left or right
    # bad_epochs_left = [
    #     x - 1
    #     for x in list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][0])
    #     + list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][0])
    # ]
    # bad_epochs_right = [
    #     x - 1
    #     for x in list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][1])
    #     + list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][1])
    # ]
    # #
    # epochs_left.drop(bad_epochs_left).load_data()
    # epochs_right.drop(bad_epochs_right).load_data()
    return raw_array_left, raw_array_right, epochs_left, epochs_right
