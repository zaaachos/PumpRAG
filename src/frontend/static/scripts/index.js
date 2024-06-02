const chatBox = document.getElementById("chat-box");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");

// Initialize the connection of user with virtual assistant via Web Sockets
const ws = new WebSocket("ws://localhost:8000/chat");

// when logged, send a msg to console (for debugging purpose)
ws.onopen = () => {
  console.log("Connected to WebSocket server");
};

// when user send message, send through sockets and append a new DIV element
sendButton.addEventListener("click", () => {
  const message = messageInput.value.trim();
  if (message !== "") {
    appendMessage(message, "sent");
    ws.send(message);
    messageInput.value = "";
    appendMessage("", "received");
  }
});

// updates the chat box with the new message
ws.onmessage = (event) => {
  const message = event.data;
  updateChat(message);
};

// updates the existing message in the chat box
function updateChat(message) {
  const messageElements = chatBox.getElementsByClassName("message");
  // Update the text content of the last message element
  const lastMessageElement = messageElements[messageElements.length - 1];
  lastMessageElement.textContent += message;

  // set maximum scroll
  chatBox.scrollTop = chatBox.scrollHeight;
}

// appends a new DIV element for each message
function appendMessage(message, type) {
  const messageContainer = document.createElement("div");
  messageContainer.classList.add("message-container");

  const messageElement = document.createElement("div");
  messageElement.textContent = message;
  messageElement.classList.add("message", type);

  messageContainer.appendChild(messageElement);
  chatBox.appendChild(messageContainer);

  // set maximum scroll
  chatBox.scrollTop = chatBox.scrollHeight;
}
