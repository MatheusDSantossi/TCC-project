import { Fragment, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import { BsInfoCircleFill } from "react-icons/bs";
import { IconButton } from "@mui/material";
import { MdOutlineClose } from "react-icons/md";

function InfoDialogScreen({ infoText }) {
  const [open, setOpen] = useState(false);
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down("sm"));

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <Fragment>
      <IconButton
        aria-label="info about the field"
        size="small"
        variant="text"
        color="primary"
        onClick={handleClickOpen}
      >
        <span className="p-1 rounded-full text-secondary">
          <BsInfoCircleFill />
        </span>
      </IconButton>
      <Dialog
        fullScreen={fullScreen}
        open={open}
        onClose={handleClose}
        aria-labelledby="responsive-dialog-title"
        sx={{
          ".MuiPaper-root": {
            backgroundColor: "#012921",
          },
        }}
      >
        <MdOutlineClose className="mt-4 ml-4 text-3xl text-white cursor-pointer hover:text-secondary" onClick={handleClose} />
        <DialogTitle
          id="responsive-dialog-title"
          className="text-2xl font-semibold text-center text-secondary"
        >
          Informações Importantes:
        </DialogTitle>
        <DialogContent sx={{ fontSize: "3rem" }}>
          <DialogContentText sx={{ color: "#FFFF" }}>
            {infoText}
          </DialogContentText>
        </DialogContent>
      </Dialog>
    </Fragment>
  );
}

export default InfoDialogScreen;
