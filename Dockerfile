# استخدم صورة أساس (مثل Python)
FROM python:3.10-slim

# أنشئ مجلد داخل الحاوية واشتغل فيه
WORKDIR /app

# انسخ كل ملفات المشروع من جهازك إلى داخل الحاوية
COPY . /app

# ثبّت المكتبات المطلوبة من requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# حدد الأمر الذي يشغل البرنامج داخل الحاوية
CMD ["python", "main.py"]
