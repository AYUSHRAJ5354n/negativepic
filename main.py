import os
import cv2
import numpy as np
import io
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler

# --- 1. HEALTH CHECK SERVER ---
# This part tells Koyeb "I am alive" so it doesn't kill the bot.
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is running smoothly!")

    def log_message(self, format, *args):
        return # Silent logs for health checks

def run_health_check_server():
    # Koyeb passes the port as an environment variable (usually 8000)
    port = int(os.getenv("PORT", 8000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logging.info(f"Health check server started on port {port}")
    server.serve_forever()

# --- 2. ADVANCED IMAGE PROCESSING ENGINE ---
def process_negative(image_bytes):
    # Load image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to float for precise math
    img_f = img.astype(np.float32)

    # AUTO-CONFIG: Neutralize the Orange Mask
    # We sample the film base (brightest parts of negative)
    for i in range(3):
        mask_val = np.percentile(img_f[:,:,i], 98)
        if mask_val > 0:
            img_f[:,:,i] = np.clip(img_f[:,:,i] * (255.0 / mask_val), 0, 255)
    
    # Invert to Positive
    positive = 255 - img_f.astype(np.uint8)

    # REAL-COLOR: White Balance (Gray World Algorithm)
    res_f = positive.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(res_f[:,:,0]), np.mean(res_f[:,:,1]), np.mean(res_f[:,:,2])
    avg_all = (avg_b + avg_g + avg_r) / 3
    res_f[:,:,0] *= (avg_all / avg_b)
    res_f[:,:,1] *= (avg_all / avg_g)
    res_f[:,:,2] *= (avg_all / avg_r)
    res = np.clip(res_f, 0, 255).astype(np.uint8)

    # ENHANCE: Adaptive Contrast (CLAHE)
    lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Encode to high-quality JPEG
    _, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()

# --- 3. BOT HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎞️ **Advanced Negative Developer Bot**\nSend me a photo of your film strip!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = await update.message.reply_text("⚙️ Developing... please wait.")
    try:
        photo_file = await update.message.photo[-1].get_file()
        img_bytes = await photo_file.download_as_bytearray()
        
        output_bytes = process_negative(img_bytes)
        
        await update.message.reply_photo(
            photo=io.BytesIO(output_bytes), 
            caption="✨ Real Colors Restored with Auto-Config"
        )
        await status.delete()
    except Exception as e:
        await status.edit_text(f"❌ Error: {str(e)}")

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # A. Start Health Check server in background thread
    threading.Thread(target=run_health_check_server, daemon=True).start()
    
    # B. Run Telegram Bot
    TOKEN = os.getenv("BOT_TOKEN")
    if not TOKEN:
        logging.error("No BOT_TOKEN provided!")
    else:
        app = ApplicationBuilder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        
        logging.info("Bot is starting polling...")
        app.run_polling()
