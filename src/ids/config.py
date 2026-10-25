"""Configuration and NSL-KDD schema."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Public GitHub mirror of the original NSL-KDD files (Canadian Institute for Cybersecurity, UNB).
DATA_URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}

FEATURE_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
ALL_COLS = FEATURE_COLS + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURE_COLS if c not in CATEGORICAL]
LOG_COLS = ["duration", "src_bytes", "dst_bytes"]  # extremely heavy-tailed -> log1p

# Standard grouping of the 39 attack names into 4 families (+ normal).
ATTACK_FAMILY = {
    "normal": "normal",
    # Denial of service
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
                          "apache2", "processtable", "udpstorm", "worm"]},
    # Probing / reconnaissance
    **{a: "Probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    # Remote-to-local (unauthorised access from a remote machine)
    **{a: "R2L" for a in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                          "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
                          "snmpguess", "xlock", "xsnoop", "httptunnel"]},
    # User-to-root (privilege escalation)
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"]},
}
CLASSES_MULTI = ["normal", "DoS", "Probe", "R2L", "U2R"]
CLASSES_BINARY = ["normal", "attack"]

SEEDS = [0, 1, 2]
VAL_FRACTION = 0.2
