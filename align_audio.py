import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

from merge import *


def read_mapped(files):
    mapped = []
    for file in files:
        try:
            f = wav.read(file, mmap=True)[1]
            mapped.append(f)
        except:
            print("Bad audio chunk:", file, "Skipping...")
    return mapped


# Audio sampling rate is hardcoded to 48 kHz - depends on radio firmware
def decode_single(path, plot=False, save=True, savefig=True, silent=False, verbose=True):
    files = sorted(glob.glob(path + "/*.wav"))
    print("\n%d files in %s:" % (len(files), path), files[:10])
    if len(files) == 0:
        return

    # mapped = [wav.read(file, mmap=True)[1] for file in files]
    mapped = read_mapped(files)
    data = np.concatenate(mapped, axis=0)  # makes a copy here?
    print("data:", data.shape, "- %.3f sec @ 48 kHz" % (data.shape[0]/48000))

    sync = data[:, -1]  # last channel contains the timestamps
    b = sync < 0  # digitize the signal
    es = np.diff(b)  # find edges
    ei = np.nonzero(es)[0]  # find edges location
    l = np.diff(ei)  # find distance between consecutive edges
    # timestamps starts after a constant for more than 200 audio samples
    e0 = ei[np.nonzero(l > 200)[0] + 1] + 5  # +1 moves to start bit and +5 skips it (5 audio samples per bit)
    e0 = e0[:-1]  # last timestamp might be incomplete
    n = e0.shape[0]  # total timestamps to decode
    off = np.tile(np.arange(32), (n, 1)) * 5 + 3  # offsets to sample 32-bits for each timestamp
    off = off + e0[:, None]  # move to zero bit positions for each timestamp
    bits = b[off.ravel()].reshape((n, 32)).astype(np.uint8)  # sample timestamp bits in the middle of 5 audio samples
    t = np.packbits(bits, axis=-1, bitorder="little")  # pack bits into bytes first
    ts = np.sum(t * np.array([1, 2 ** 8, 2 ** 16, 2 ** 24])[None, :], axis=1)  # now pack bytes into timestamps
    idx = np.nonzero(ts > 20_000_000)[0]  # Buggy unrealistic timestamps due to the CRC check failure in radio receiver
    if idx.shape[0] > 0:
        print("Found unrealistic timestamps:", ts[idx])
        ts[idx] += np.arange(idx.shape[0])  # Force-trigger a gap to have them removed
    err = np.nonzero(np.diff(ts) - 10)[0]  # detect decoding errors (due to the overrun errors)

    if not silent:
        print(len(e0), "audio:", e0[:10])
        print(len(ts), "radio:", ts[:10])
        if len(err):
            print(len(err), "errors:", err)
            if verbose:
                print("%3s %9s %15s %15s %15s %15s" % ("i", "ri", "ti", "t1", "t1 - ti", "ri - r0"))
                for i in range(len(err)):
                    r0, ri = 0 if i == 0 else err[i - 1], err[i]
                    ti, t1 = ts[err[i]], ts[err[i] + 1]
                    print("%3d %9d %15d %15d %15d %15d" % (i, ri, ti, t1, t1 - ti, ri - r0))

    e0, ts = np.delete(e0, err), np.delete(ts, err)  # delete bad timestamps

    over = np.nonzero(np.diff(ts) - 10)[0]  # detect overruns
    jumps, total_overrun = [], 0

    if len(over):
        if not silent:
            print(len(over), "overruns:", over)
            if verbose:
                print("%3s %9s %15s %9s %15s %9s %9s %17s" %
                      ("i", "oi", "pi", "ti", "p1", "t1", "p1-pi", "(t1 - ti) * 40"))
        for i in range(len(over)):
            pi, p1 = e0[over[i]], e0[over[i] + 1]
            ti, t1 = ts[over[i]], ts[over[i] + 1]
            overrun = (t1-ti) * 40 - (p1-pi)
            jumps.append((pi, ti, p1, t1, overrun, (t1-ti) * 40))
            total_overrun += overrun
            if not silent and verbose:
                print("%3d %9d %15d %9d %15d %9d %9d %17d" % (i, over[i], pi, ti, p1, t1, p1-pi, (t1-ti) * 40))
        print("total_overrun: %d audio samples (%.3f sec - %.2f %%)" %
              (total_overrun, total_overrun / 48000, 100 * total_overrun / data.shape[0]))

    timestamps, jumps = np.stack((e0, ts), axis=1), np.array(jumps)  # convert to numpy for convenience

    # find clock frequency relative to the radio
    ref = np.concatenate([[0], over, [ts.shape[0] - 1]])  # include beginning and the end
    im = np.argmax(np.diff(ref))  # find longest continuous range for better accuracy fit
    l, r = ref[im] + 2, ref[im + 1]  # skip jump (if any) and extra timestamp in the beginning
    x, y = e0[l:r], ts[l:r]  # and skip one timestamp in the end of the range
    mb = np.polyfit(x, y, 1)  # fit a line
    clock_adjustment = (mb[0] - 0.025) / 0.025  # 400 audio samples per 10 timestamps @ 1200 Hz radio refresh rate

    if not silent:
        print("reference points:", ref)
        print("max range: %d %d (%d)" % (l, r, r-l))
    print("clock adjustment:", clock_adjustment, "(>0 means radio is faster)")

    if save:
        meta = {"timestamps": timestamps.tolist(),
                "jumps": jumps.tolist(),
                "total_overrun": int(total_overrun),
                "clock_adjustment": clock_adjustment,
                "timestamps_help": "[audio_position, radio_timestamp]",
                "jumps_help": "[audio_start, radio_start, audio_finish, radio_finish,"
                              " audio_samples_overrun, audio_samples_to_skip]",
                "total_overrun_help": "audio_samples",
                "clock_adjustment_help": ">0 means radio is faster"}

        with open(path + "/timestamps.json", "w") as f:
            json.dump(meta, f, indent=4)

    if plot:
        plt.figure(path, (16, 9))
        plt.plot(timestamps[:, 0], timestamps[:, 1], "b-", label="All timestamps")
        plt.plot(x, np.poly1d(mb)(x), "r-", label="Clock fit")
        for i in range(jumps.shape[0]):
            plt.plot((jumps[i, [0, 0]] + [0, jumps[i, 5]]), jumps[i, [1, 3]], ".-", label="Jump %d" % i)
        plt.xlabel("Audio position")
        plt.ylabel("Radio timestamp")
        plt.title("Clock adjustment = " + str(clock_adjustment) + " (>0 means radio is faster)")
        plt.legend()
        plt.tight_layout()

        if savefig:
            plt.savefig(path + "/../array.png", dpi=120)


def load_data(path):
    files = sorted(glob.glob(path + "/*.wav"))
    if not len(files):
        return None

    # mapped = [wav.read(file, mmap=True)[1] for file in files]
    mapped = read_mapped(files)
    return np.concatenate(mapped, axis=0)  # makes a copy here?


def load_meta(path):
    filename = path + "/timestamps.json"
    if not os.path.isfile(filename):
        return None

    with open(filename, "r") as f:
        meta = json.load(f)
        meta["timestamps"] = np.array(meta["timestamps"])
        return meta


def load_all(base_path):
    data, meta = [None] * 4, [None] * 4
    for i in range(4):
        path = base_path % (i + 1)
        data[i] = load_data(path)
        meta[i] = load_meta(path)
    return data, meta


def align_all(session):
    data, meta = load_all(session + "/sensor_%d/audio")

    timestamps = [m["timestamps"] if m is not None else None for m in meta]
    min_t = min([t[0, 1] for t in timestamps if t is not None])
    max_t = max([t[-1, 1] for t in timestamps if t is not None])
    print("common range: %d %d (%d - %.3f sec @ 48 kHz)" %
          (min_t, max_t, max_t - min_t, (max_t - min_t) * 40 / 48000))

    if max_t < min_t:
        print("\tdiscarded")
        return

    for i, (d, t) in enumerate(zip(data, timestamps)):
        if d is not None and t is not None:
            merged = np.zeros(((max_t - min_t + 10) * 40, 12), dtype=np.int16)
            print(i+1, merged.shape, "from", d.shape)
            for j in range(t.shape[0]):
                ap, rt = t[j, :]
                if rt < min_t or rt > max_t:
                    continue
                new_ap = (rt - min_t) * 40
                chunk = 400 if rt == max_t else 401
                merged[new_ap:new_ap+chunk, :] = d[ap:ap+chunk, :12]

            wav.write(session + "/sensor_%d/array.wav" % (i + 1), 48000, merged)
            with open(session + "/sensor_%d/array.json" % (i + 1), "w") as f:
                json.dump({"start": min_t, "stop": max_t,
                           "radio_period": 10, "audio_period": 400,
                           "radio_frequency, Hz": 1200,
                           "audio_frequency, Hz": 48000}, f, indent=4, cls=NumpyEncoder)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "serif"


def display_all(base_path, channel=0, title=None, savefig=None):
    print("\ndisplay:", base_path)
    data = [None] * 4
    for i in range(4):
        filename = base_path % (i + 1)
        if not os.path.isfile(filename):
            continue
        print(filename)
        data[i] = wav.read(filename, mmap=True)[1]

    plt.figure(base_path, (16, 9))
    stride = 100
    for i, d in enumerate(data):
        if d is None:
            continue
        x = np.arange(d[::stride, channel].shape[0]) / (60 * 48000/stride)
        plt.plot(x, d[::stride, channel] + (-i-1) * 1000, label="Sensor " + str(i+1))
        if i == 2:
            next(plt.gca()._get_lines.prop_cycler)

    plt.xlim([-0.02, 0.02 + x.shape[0] / (60 * 48000/stride)])
    plt.ylim([-5500, 600])
    plt.xlabel("Time, minutes")
    # plt.ylabel("Amplitude, 16-bit (with -SensorID*1000 offset)")
    plt.ylabel("Amplitude, 16-bit (with offset)")
    if title is None:
        plt.title("Channel %d of each sensor's microphone array" % channel)
    else:
        if title != "":
            plt.title(title)
    plt.legend()
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig + "/audio.png", dpi=240)


def crop_single(path, title, channels=(0, 3), offset=173):
    print("Using offset %d for" % offset, path)
    data = wav.read(path + "array.wav", mmap=True)[1]
    meta = json.load(open(path + "array.json"))
    vmeta = json.load(open(path + "left.json"))

    al, ar = meta["start"] + offset, meta["stop"] + offset
    vl, vr = vmeta["start"], vmeta["stop"]
    a_per, r_per, v_per = meta["audio_period"], meta["radio_period"], vmeta["period"]
    r2a = meta["audio_frequency, Hz"] // meta["radio_frequency, Hz"]

    ol = ar - al + r_per
    nl = vr - vl + round(v_per)
    new_data = np.zeros((nl * r2a, len(channels)), dtype=np.int16)

    if al < vl:
        l_from = (vl-al) * r2a
        l_to = 0

        if nl < ol - l_from:
            l = nl * r2a
        else:
            l = (ol - l_from) * r2a
    else:
        l_from = 0
        l_to = (al-vl) * r2a

        if ol < nl - l_to:
            l = ol * r2a
        else:
            l = (nl - l_to) * r2a

    new_data[l_to:l_to+l, :] = data[l_from:l_from+l, channels]

    wav.write(path + "../%s.wav" % title, 48000, new_data)


if __name__ == '__main__':
    data_path = '../data/'
    sessions = glob.glob(data_path + "*/")  # glob.glob behaves differently outside of __main__ (i.e. inside functions)

    for session in browse_sessions(sessions):
        # for i in range(4):
        #     audio = session + "sensor_%d/" % (i + 1) + "audio/"
        #     decode_single(audio, plot=True, save=True, silent=False, verbose=True)

        # align_all(session)
        display_all(session + "/sensor_%d/array.wav", savefig=session, title="")

        # for i in range(4):
        #     crop_single(session + "sensor_%d/" % (i + 1), "stereo_%d" % (i + 1))

    plt.show()
