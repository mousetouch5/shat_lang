const express = require("express");
const path = require("path");

const app = express();
const PORT = 3000;

// Serve browser app
app.use(express.static(path.join(__dirname, "tm-browser")));

// IMPORTANT: listen on 0.0.0.0 (all interfaces)
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on:`);
  console.log(`  http://localhost:${PORT}`);
  console.log(`  http://<YOUR-IP>:${PORT}`);
});
