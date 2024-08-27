"use client";
import { TextField, Box, Stack, Button, IconButton } from "@mui/material";
import { useState, useRef } from "react";
import MicIcon from "@mui/icons-material/Mic";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import StopIcon from "@mui/icons-material/Stop";

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello Aggie! I'm here to help you find Computer Science professors. How can I help you today?"
    }
  ]);
  const [message, setMessage] = useState('');
  const [speaking, setSpeaking] = useState(false);
  const speechSynthesisRef = useRef(null);

  const sendMessage = async () => {
    
    setMessages((messages) => [
      ...messages,
      { role: "user", content: message },
      { role: "assistant", content: '' }
    ])

    

    const response = fetch('/api/chat', {
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify([...messages, { role: "user", content: message }]),
    }).then(async (res) => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let result = '';
      return reader.read().then(function processText({ done, value }) {
        if (done) {
          return result
        }
        const text = decoder.decode(value || new Uint8Array(), { stream: true });
        setMessages((messages) => {
          let lastMessage = messages[messages.length - 1];
          let otherMessages = messages.slice(0, messages.length - 1);
          return [
            ...otherMessages,
            { ...lastMessage, content: lastMessage.content + text }
          ];
        });

        return reader.read().then(processText);
      });
      
    });
  };

  const startSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.onstart = () => {
      console.log('Voice recognition started. Speak into the microphone.');
    };

    recognition.onspeechend = () => {
      recognition.stop();
      console.log('Voice recognition ended.');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setMessage(transcript); // Set recognized text as the input message
    };

    recognition.start();
  };

  const speak = (text) => {
    const speech = new SpeechSynthesisUtterance();
    speech.text = text;
    speech.volume = 1;
    speech.rate = 1;
    speech.pitch = 1;
    setSpeaking(true);
    speech.onend = () => setSpeaking(false);
    speechSynthesisRef.current = speech;
    window.speechSynthesis.speak(speech);
  };

  const stopSpeaking = () => {
    if (speechSynthesisRef.current) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
    }
  };

  return (
    <Box
      width="100vw"
      height="100vh"
      display="flex"
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
      sx={{
        backgroundImage: "/ucdavis.png",
        background: "linear-gradient(to right, #1e3c72, #2a5298)",
        fontFamily: "'Roboto', sans-serif",
        color: "#fff",
        padding: "20px", // Add padding to ensure content doesn’t touch screen edges
        boxSizing: "border-box",
      }}
    >
      <Box
        width="100%"
        maxWidth="600px"  // Constrain max width for responsiveness
        height="100%"
        maxHeight="90vh"  // Set max height relative to viewport height
        borderRadius="12px"
        boxShadow="0 4px 12px rgba(0, 0, 0, 0.1)"
        bgcolor="#ffffff"
        p={2}
        sx={{
          backdropFilter: "blur(10px)",
          background: "rgba(255, 255, 255, 0.8)",
          overflow: "hidden", // Prevent overflow of content
        }}
      >
        <Stack
          direction="column"
          spacing={2}
          height="100%"
        >
          <Box
            p={2}
            bgcolor="#1e3c72"
            color="#fff"
            borderRadius="12px"
            textAlign="center"
            fontSize="24px"
            fontWeight="bold"
          >
            Aggie AI <img src="/ucdavis.png" alt="UC Davis" style={{ width: '50px', marginRight: '10px', verticalAlign: 'middle' }} /> CS Advising 
          </Box>
          <Stack
            direction="column"
            spacing={2}
            flexGrow={1}
            overflow='auto'
            maxHeight='100%'
          >
            {messages.map((message, index) => (
              <Box
                key={index}
                display="flex"
                justifyContent={message.role === "assistant" ? 'flex-start' : 'flex-end'}
              >
                <Box
                  bgcolor={message.role === 'assistant' ? "#4a90e2" : "#FBC02D"}
                  color="white"
                  borderRadius={16}
                  p={3}
                  sx={{
                    boxShadow: "0 2px 6px rgba(0, 0, 0, 0.1)",
                    maxWidth: "75%"
                  }}
                >
                  {message.content}
                </Box>
              </Box>
            ))}
          </Stack>
          <Stack direction="row" spacing={2}>
            <TextField
              label="Message"
              fullWidth
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              variant="outlined"
              sx={{
                bgcolor: "rgba(255, 255, 255, 0.8)",
                borderRadius: "8px",
                fontSize: "16px",
                "& .MuiOutlinedInput-root": {
                  "& fieldset": {
                    borderColor: "#1e3c72"
                  },
                  "&:hover fieldset": {
                    borderColor: "#2a5298"
                  },
                  "&.Mui-focused fieldset": {
                    borderColor: "#2a5298"
                  }
                }
              }}
            />
            <Button
              variant='contained'
              onClick={sendMessage}
              sx={{
                backgroundColor: "#4a90e2",
                "&:hover": {
                  backgroundColor: "#2a5298"
                }
              }}
            >
              Send
            </Button>
            <IconButton
              color="primary"
              onClick={startSpeechRecognition}
              sx={{
                bgcolor: "#1e3c72",
                color: "#fff",
                "&:hover": {
                  bgcolor: "#5e35b1"
                }
              }}
            >
              <MicIcon />
            </IconButton>
            {speaking ? (
              <IconButton
                color="secondary"
                onClick={stopSpeaking}
                sx={{
                  bgcolor: "#f44336",
                  color: "#fff",
                  "&:hover": {
                    bgcolor: "#d32f2f"
                  }
                }}
              >
                <StopIcon />
              </IconButton>
            ) : (
              <IconButton
                color="secondary"
                onClick={() => speak(messages[messages.length - 1].content)}
                sx={{
                  bgcolor: "#ffb74d",
                  color: "#fff",
                  "&:hover": {
                    bgcolor: "#ffa726"
                  }
                }}
              >
                <VolumeUpIcon />
              </IconButton>
            )}
          </Stack>
        </Stack>
      </Box>
    </Box>
  );
}
