import os
import shutil

# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import Flask, jsonify, request, redirect, url_for, render_template

# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
import setproctitle

setproctitle.setproctitle("WebApp")

app = Flask(__name__)

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = "./static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])


def allwed_file(filename):
	# .があるかどうかのチェックと、拡張子の確認
	# OKなら１、だめなら0
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ファイルを受け取る方法の指定
@app.route("/", methods=["GET", "POST"])
def upload_file():
	# リクエストがポストかどうかの判別
	if request.method == "POST":
		# ファイルがなかった場合の処理
		if "file" not in request.files:
			print("ファイルがありません")
			return redirect(request.url)
		# データの取り出し
		file = request.files["file"]
		# ファイル名がなかった時の処理
		if file.filename == "":
			print("ファイルがありません")
			return redirect(request.url)
		# ファイルのチェック
		if file and allwed_file(file.filename):
			# 危険な文字を削除（サニタイズ処理）
			filename = secure_filename(file.filename)
			# ファイルの保存
			file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
			# アップロード後のページに転送
			return redirect(url_for("uploaded_file", filename=filename))
		else:
			return render_template("filename_notAllowed.html", extensions=list(ALLOWED_EXTENSIONS))
	return render_template("index.html")


# アップロードされた画像ファイルに書かれた数字を分類
from PIL import Image
import datetime
from imageClassifier import ImageClassifier

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
classifier = ImageClassifier()


@app.route("/upload/<filename>")
def uploaded_file(filename):
	path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
	image = Image.open(path)
	pred = classifier.predict(image)
	filenameOrig = filename
	now = datetime.datetime.now(JST)
	filename = now.strftime("%Y%m%d%H%M%S_") + filename
	shutil.move(path, os.path.join(app.config["UPLOAD_FOLDER"], filename))
	return render_template(
		"result.html",
		prediction=pred,
		filename=url_for("static", filename=filename),
		filenameOrig=filenameOrig,
	)


# 分類結果についてのフィードバックを受け取り、保存する
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

s3 = boto3.resource(
		service_name          = "s3",
		endpoint_url          = "http://"+os.getenv('STORAGE_HOST')+":9000",
		aws_access_key_id     = os.getenv('MINIO_ROOT_USER'),
		aws_secret_access_key = os.getenv('MINIO_ROOT_PASSWORD'),
		config                = Config(proxies={'http':  os.getenv('HTTP_PROXY'),
									   			'https': os.getenv('HTTPS_PROXY')})
)
try:
	bucket = s3.create_bucket(Bucket="digit-images")
except ClientError as e:
	if e.response["Error"]["Code"] in (
		"BucketAlreadyExists",
		"BucketAlreadyOwnedByYou",
	):
		bucket = s3.Bucket("digit-images")
	else:
		print(f"Unknown exception.\n\t " + e.response["Error"]["Code"])
		raise

import re
import MySQLdb

mysql = MySQLdb.connect(
		user     = os.getenv('MYSQL_USER'),
		password = os.getenv('MYSQL_PASSWORD'),
		database = os.getenv('MYSQL_DATABASE'),
		host     = os.getenv('DATABASE_HOST'),
		port     = 3306
)
mysqlCursor = mysql.cursor()
tableName = "uploaded"
mysqlCursor.execute(
	f"create table IF NOT EXISTS {tableName}(id INT AUTO_INCREMENT primary key, relpath varchar(100), label INT, date DATETIME, is_used BOOLEAN)"
)


@app.route("/feedback", methods=["POST"])
def get_feedback():
	feedback = int(request.form.get("feedback"))
	filename = request.form.get("filename")
	if feedback is None:
		feedback = "NULL"
	basename = os.path.basename(filename)
	filepath = os.path.join(app.config["UPLOAD_FOLDER"], basename)

	datetime = re.search(r"\d+", basename).group()
	mysqlCursor.execute(
		f"INSERT INTO {tableName}(relpath,label,date,is_used) VALUES('{basename}',{feedback},'{datetime}',false)"
	)
	mysql.commit()

	obj_key  = tableName+'/'+basename
	bucket.upload_file(f"{filepath}", obj_key)
	objs = list(bucket.objects.filter(Prefix=obj_key))
	if len(objs) > 0 and objs[0].key == obj_key:
		os.remove(filepath)
	return render_template("got_feedback.html", feedback=feedback)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
