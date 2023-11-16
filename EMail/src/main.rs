mod imap;
use druid::{Data,widget::{Label,Flex},Env,Widget,WindowDesc,AppLauncher,LocalizedString};

#[derive(Clone, Data)]
struct MailData {   
    num: u32,
    txt: String,
}

fn ui_builder() -> impl Widget<MailData> {
    let label  = Label::new(|_data: &MailData, _: &Env| format!("Mail Content:"));
    let txt_label  = Label::new(|_data: &MailData, _: &Env| format!("{}",_data.txt));
    Flex::column()
        .with_child(label)
        .with_child(Flex::row()
        .with_child(txt_label))
}

fn main() {
    let test_txt: String = imap::fetch_inbox_top().unwrap().unwrap().to_string();
    //let substring = &test_txt[3100..3500];
    let main_window = WindowDesc::new(ui_builder())
        .title(LocalizedString::new("Rust Mail GUI Example"));
        
    AppLauncher::with_window(main_window)
        .log_to_console()
        .launch(MailData {num:0,txt:test_txt.to_string()}).unwrap();
}
