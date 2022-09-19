import os
import subprocess

# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import Flask, request, redirect, url_for

# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
import setproctitle

setproctitle.setproctitle("webApp_builder")
baseDir=os.path.dirname(__file__)

app = Flask(__name__)

# ファイルのアップロード先のディレクトリ
UPLOAD_FOLDER = os.path.join(baseDir, "trainedDNN")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# アップロードされる拡張子の制限
ALLOWED_FILES = set(["imageClassifier.py", "model_definition.py", "model_weights.pth"])


def allwed_file(filename):
	# ファイル名の確認
	# OKなら１、だめなら0
	return filename in ALLOWED_FILES

@app.route("/", methods=["GET"])
def status():
	return {"builder_status": 'ready'}

# ファイルを受け取る方法の指定
@app.route("/", methods=["POST"])
def upload_file():
	error = []
	got_all_files = True
	for filename in ALLOWED_FILES:
		if filename in request.files:
			file = request.files[filename]
			file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
		else:
			str=f"Counld not receive \"{filename}\""
			error.append(str)
			print(str)
			got_all_files = False
	if not got_all_files:
		return error

	build_result = subprocess.run( [os.path.join(baseDir, "build_new_image.sh"), app.config["UPLOAD_FOLDER"]])

	# build_result = subprocess.Popen([os.path.join(baseDir, "build_new_image.sh"), app.config["UPLOAD_FOLDER"]],
	# 					stdout=subprocess.PIPE,
	# 					stderr=subprocess.PIPE)
	return { "builder_returncode": build_result.returncode }

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
