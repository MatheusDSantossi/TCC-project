import React, { useState, useEffect } from "react";
import { Snackbar, Alert } from "@mui/material";
import PropTypes from "prop-types";

/**
 * CustomMessage wraps MUI Snackbar + Alert for transient messages.
 *
 * Props:
 * - open: boolean, controls visibility
 * - onClose: function, called when message closes
 * - severity: one of 'success' | 'info' | 'warning' | 'error'
 * - message: string
 * - duration: autoClose time in ms
 */
const CustomMessage = ({
  open,
  onClose,
  severity = "success",
  message,
  duration = 2000,
}) => {
  return (
    <Snackbar
      open={open}
      autoHideDuration={duration}
      onClose={onClose}
      anchorOrigin={{ vertical: "top", horizontal: "right" }}
    >
      <Alert
        elevation={6}
        variant="filled"
        onClose={onClose}
        severity={severity}
        sx={{ width: "100%" }}
      >
        {message}
      </Alert>
    </Snackbar>
  );
};

CustomMessage.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  severity: PropTypes.oneOf(["success", "info", "warning", "error"]),
  message: PropTypes.string.isRequired,
  duration: PropTypes.number,
};

export default CustomMessage;
