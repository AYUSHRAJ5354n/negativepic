import os
import cv2
import numpy as np
import io
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler

# Logging
logging.basicConfig(level=logging.INFO)

def process_advanced_negative(image_bytes):
    # 1. Load Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. AUTO-CONFIG: Remove Orange Mask
    # We sample the 97th percentile of the image. In your photo, 
    # this represents the unexposed orange film base.
    img_f = img.astype(np.float32)
    for i in range(3):
        mask_val = np.percentile(img_f[:,:,i], 97)
        img_f[:,:,i] = np.clip(img_f[:,:,i] * (255.0 / mask_val), 0, 255)
    
    # 3. INVERT to Positive
    positive = 255 - img_f.astype(np.uint8)

    # 4. COLOR BALANCING (Real Color Restoration)
    # We balance the channels so whites look white.
    result = positive.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result[:,:,0]), np.mean(result[:,:,1]), np.mean(result[:,:,2])
    avg_all = (avg_b + avg_g + avg_r) / 3
    result[:,:,0] *= (avg_all / avg_b)
    result[:,:,1] *= (avg_all / avg_g)
    result[:,:,2] *= (avg_all / avg_r)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 5. ENHANCE CONTRAST (CLAHE)
    # This brings out details in faces and shadows
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Output
    _, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎞️ **Advanced Film Bot Ready**\nSend me your film strip photos!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = await update.message.reply_text("⚙️ Auto-Configuring Colors...")
    try:
        photo = await update.message.photo[-1].get_file()
        img_bytes = await photo.download_as_bytearray()
        
        output = process_advanced_negative(img_bytes)
        
        await update.message.reply_photo(photo=io.BytesIO(output), caption="✨ Developed with Real Colors")
        await status.delete()
    except Exception as e:
        await status.edit_text(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()
