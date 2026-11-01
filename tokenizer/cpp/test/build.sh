#!/bin/sh
# tokenizer/cpp/test/build.sh -- the test binaries of tokenizer/cpp/json.cpp, built in
# build/tokenizer/cpp/test/ (B below), with JSONTestSuite cloned there once.
#
#     sh tokenizer/cpp/test/build.sh                    build them
#     python3 tokenizer/cpp/test/tests.py               every section; exits 1 on a failure
#     DRIVER=driver_asan python3 tokenizer/cpp/test/tests.py    the same under ASan and UBSan
#     DRIVER=driver_clang python3 tokenizer/cpp/test/tests.py   the same with the clang build
#     B/unit                                            the C++ interface
#     python3 tokenizer/cpp/test/mutate.py              18 deliberate bugs; every one must be caught
#     B/fuzz -max_total_time=600 B/corpus B/JSONTestSuite/test_parsing
#     sh tokenizer/cpp/test/build.sh coverage           line and branch coverage of json.cpp (clang)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
BIN="$(cd "$HERE/../../.." && pwd)/build/tokenizer/cpp/test"
mkdir -p "$BIN" "$BIN/corpus"
[ -d "$BIN/JSONTestSuite" ] || git clone -q --depth 1 https://github.com/nst/JSONTestSuite "$BIN/JSONTestSuite"
W="-std=c++17 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror"
J="$SRC/json.cpp $SRC/utf8.cpp"
g++ $W -O2 -I"$SRC" "$HERE/driver.cpp" $J -o "$BIN/driver"
g++ $W -O2 -I"$SRC" "$HERE/unit.cpp" $J -o "$BIN/unit"
SAN="-std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
g++ $SAN -I"$SRC" "$HERE/driver.cpp" $J -o "$BIN/driver_asan"
g++ $SAN -I"$SRC" "$HERE/unit.cpp" $J -o "$BIN/unit_asan"
if command -v clang++ >/dev/null; then
  clang++ $W -O2 -I"$SRC" "$HERE/driver.cpp" $J -o "$BIN/driver_clang"
  clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -I"$SRC" \
    "$HERE/fuzz.cpp" $J -o "$BIN/fuzz" || echo "[json test] no libFuzzer runtime; fuzz not built"
fi
if [ "$1" = coverage ]; then
  C="-std=c++17 -O0 -fprofile-instr-generate -fcoverage-mapping -I$SRC"
  clang++ $C "$HERE/driver.cpp" $J -o "$BIN/driver_cov"
  clang++ $C "$HERE/unit.cpp" $J -o "$BIN/unit_cov"
  rm -f "$BIN"/*.profraw
  LLVM_PROFILE_FILE="$BIN/p-%p.profraw" DRIVER=driver_cov SCALE=0.25 python3 "$HERE/tests.py" >/dev/null
  LLVM_PROFILE_FILE="$BIN/u.profraw" "$BIN/unit_cov" >/dev/null
  llvm-profdata merge -o "$BIN/all.profdata" "$BIN"/*.profraw
  llvm-cov report "$BIN/driver_cov" -object "$BIN/unit_cov" -instr-profile="$BIN/all.profdata" \
    "$SRC/json.cpp" -show-branch-summary
fi
echo "[json test] built in $BIN"
