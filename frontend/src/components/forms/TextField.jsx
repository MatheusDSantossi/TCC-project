import * as React from 'react';
import '../../App.css'
import TextField from '@mui/material/TextField';
import {Controller} from 'react-hook-form'

export default function CustomTextField(props) {
  const {label, name, control} = props
  return (

     <Controller
        name = {name}
        control = {control}
        render = {({
            field:{onChange, value}, 
            fieldState : {error},
            formState,
        }) =>(

          <TextField 
          id="outlined-basic" 
          onChange = {onChange}
          value = {value}
          label= {label}
          variant="outlined" 
          className={"myForm"}
          error = {!!error}
          helperText = {error?.message}
          sx={{ padding: "0.5rem", marginTop:"0.25rem",borderRadius: "0.5rem", borderColor: "#D1D5DB", width: "100%" }}
           />
          
        )
      }
     
     />

  );
}
