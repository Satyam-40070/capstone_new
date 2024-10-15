import express from 'express';
const app = express();
import router from './routes/routes.js';
import DBConnection from './database/db.js';

app.use(express.json());
app.use(express.urlencoded({extended: true}));

DBConnection();
app.use('/', router);

app.get('/', (req,res)=>{
    res.send('Watermark your Images');
});

app.listen(4000, ()=>{
    console.log('Server is running on port 4000');
})