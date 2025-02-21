import { Fragment, useState } from "react";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import { BsInfoCircleFill } from "react-icons/bs";

function InfoDialogScreen({infoText}) {
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
      <Button variant="" color="primary" onClick={handleClickOpen}>
        <span className="p-1 rounded-full text-secondary">
          <BsInfoCircleFill />
        </span>
      </Button>
      <Dialog
        fullScreen={fullScreen}
        open={open}
        onClose={handleClose}
        aria-labelledby="responsive-dialog-title">
            <DialogTitle id="responsive-dialog-title" className="text-2xl font-semibold text-center text-primary">
              Informações Importantes:
            </DialogTitle>
            <DialogContent>
                <DialogContentText sx={{color: "black"}}>
                    {infoText}
                </DialogContentText>
            </DialogContent>
        </Dialog>
    </Fragment>
  );
}

export default InfoDialogScreen;
