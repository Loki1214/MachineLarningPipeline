import os
import shutil
# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import Flask, request, redirect, url_for, render_template
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
def allwed_file(filename):
	# .があるかどうかのチェックと、拡張子の確認
	# OKなら１、だめなら0
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	# リクエストがポストかどうかの判別
	if request.method == 'POST':
		# ファイルがなかった場合の処理
		if 'file' not in request.files:
			print('ファイルがありません')
			return redirect(request.url)
		# データの取り出し
		file = request.files['file']
		# ファイル名がなかった時の処理
		if file.filename == '':
			print('ファイルがありません')
			return redirect(request.url)
		# ファイルのチェック
		if file and allwed_file(file.filename):
			# 危険な文字を削除（サニタイズ処理）
			filename = secure_filename(file.filename)
			# ファイルの保存
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			# アップロード後のページに転送
			return redirect(url_for('uploaded_file', filename=filename))
	return render_template('index.html')


import re
import MySQLdb
mysql = MySQLdb.connect(
		user='root',
		password='root_password',
		host='mysql',
		port=3306,
		database='DigitImages'
	)
mysqlCursor = mysql.cursor()
mysqlCursor.execute('create table IF NOT EXISTS uploadedImages(id INT AUTO_INCREMENT primary key, path varchar(100), label INT, date DATETIME, is_used BOOLEAN)')
@app.route('/feedback', methods=['POST'])
def get_feedback():
	feedback = int(request.form.get('feedback'))
	filename = request.form.get('filename')
	# if feedback is None:
	# 	dir = os.path.join(app.config['UPLOAD_FOLDER'], 'nolabel/')
	# else:
	# 	dir = os.path.join(app.config['UPLOAD_FOLDER'], str(feedback) + '/')
	# os.makedirs(dir, exist_ok=True)
	basename = os.path.basename(filename)
	# shutil.move(os.path.join(app.config['UPLOAD_FOLDER'], basename), os.path.join(dir, basename))

	datetime = re.search(r'\d+', basename).group()
	print(f"INSERT INTO uploadedImages VALUES('{basename}',{feedback},{datetime})")
	mysqlCursor.execute(f"INSERT INTO uploadedImages(path,label,date,is_used) VALUES('{basename}',{feedback},'{datetime}',false)")
	mysql.commit()
	return render_template('got_feedback.html', feedback=feedback)

from PIL import Image
import datetime
from imageClassifier import ImageClassifier

t_delta = datetime.timedelta(hours=9)
JST     = datetime.timezone(t_delta, 'JST')
now     = datetime.datetime.now(JST)
classifier = ImageClassifier()

@app.route('/upload/<filename>')
# アップロードされた画像ファイルに書かれた数字を分類
def uploaded_file(filename):
	path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	image = Image.open(path)
	pred  = classifier.predict(image)
	filenameOrig = filename
	filename = now.strftime('%Y%m%d%H%M%S_') + filename
	shutil.move(path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
	return render_template('result.html', prediction=pred, filename=url_for('static', filename=filename), filenameOrig=filenameOrig)

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)