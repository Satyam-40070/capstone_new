import express from 'express';
const router = express.Router();
import { Register } from '../controller/registerControl.js';
import { Login } from '../controller/loginControl.js'; 

router.post('/register', Register);
router.post('/login', Login);

export default router;