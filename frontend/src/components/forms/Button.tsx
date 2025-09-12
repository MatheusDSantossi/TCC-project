import * as React from 'react';
import Button from '@mui/material/Button';

export default function CustomButton(props) {
  const {label,type} = props
  return (
      <Button 
        type={type} 
        variant="contained" 
        className={"myButton"}
        sx={{ backgroundColor: 'yellow', color: 'black', fontWeight: 'bold', ":hover": { backgroundColor: "#FCD34D" }  }}
        >
            {label}
      </Button>

  );
}
