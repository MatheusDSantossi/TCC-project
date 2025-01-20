import React, { useState, useEffect } from "react";

const Typewriter = ({ text, delay, onTypingEnd }) => {
  const [currentText, setCurrentText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setCurrentText((prevText) => prevText + text[currentIndex]);
        setCurrentIndex((prevIndex) => prevIndex + 1);
      }, delay);

      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && onTypingEnd) {
      // console.log(`Typing finished, with index: ${currentIndex} and text length: ${text.length}`);
      // setHasFinished(true);
      // Triggering callback when typing finishes
      console.log("Typing finished, calling onTypingEnd...");
      onTypingEnd();
    }
  }, [currentIndex, delay, text, onTypingEnd]);

  return <span className="text-right">{currentText}</span>;
};

export default Typewriter;
