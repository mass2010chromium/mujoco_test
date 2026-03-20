SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

uv pip install pycocotools decord scikit-image scikit-learn

cd $SCRIPT_DIR/sam3
ln -sT ../sam3_weights weights

cd $SCRIPT_DIR/decord
git apply ../decord_patch.diff
